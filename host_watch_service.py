"""调宿主（N.E.K.O 本体）的"一起看视频"流水线。

宿主的 ``main_logic/watch_together`` 是她真正"看视频"的实现：先按项目自己的
判定条件挑一条（单 P、三分钟以内、每分钟弹幕 > 100），再把画面抽帧做视觉
分析、抽热点，最后按人设生成她看完的反应台词、合成语音。插件这边的刷 B 站
要的就是这套东西——它比"抓几条弹幕样本"扎实得多。

## 为什么是同步 urllib 而不是 httpx/aiohttp（2026-11-15 实测定稿）

插件跑在宿主的内嵌 Python 里。往 ``127.0.0.1:48911``（宿主自己的主服务）发
请求时，httpx 和 aiohttp 都会出现"TCP 建得上、响应永远不来、连超时都不触发"
的悬空状态，整轮刷取就这样无声无息挂死；改成进程内 ``import main_logic...``
直接 await 也一样悬空（宿主协程挂在另一个事件循环上）。

同目录的 hoi4_advisor 插件用**同步 urllib 丢进线程池**打同一个端口，一直是
通的。照这个写法改完之后一次通。所以这一层就用 ``urllib.request`` +
``asyncio.to_thread``，别改回异步 HTTP 库。

## 几条硬事实

1. ``prepare`` 同一时刻只允许一个任务，撞上了会 4xx——这是正常排队语义，
   调用方重试下一轮即可，别当错误炸出去。
2. 免费 TTS 并发满的时候 ``prepare`` 会卡在合成阶段直到宿主自己的 1800 秒
   deadline：实测一条吃满 240 秒超时。所以超时之后要熔断，本轮剩下的直接
   退回自抓弹幕，不要每条都等一遍。
3. ``discover`` 的判定条件很硬，不是每次都有货，返回 ``video: null`` 很正常。
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Iterable

#: 宿主主服务地址。插件跑在宿主进程里，所以永远是本机；留环境变量是为了
#: 换端口/换机器时不用改代码。
_DEFAULT_BASE = "http://127.0.0.1:48911"

#: discover 在宿主侧要跑一次搜索 + 一次视频详情，实测 0.4~2.1 秒。给 8 秒
#: 是留足翻页余量又不至于把一轮拖死——一轮要问十来次。
_DISCOVER_TIMEOUT = 8.0

#: prepare 的总时限（宿主自己的 deadline 是 1800 秒，这里只是别把插件循环拖死）。
_DEFAULT_PREPARE_TIMEOUT = 150.0

#: 轮询间隔。宿主的阶段推进粒度是十几秒，3 秒够细了。
_POLL_INTERVAL = 3.0

#: 反应台词会出现在多个字段名里，这几个是见过的写法。
_TEXT_KEYS = ("text", "line", "speech", "reaction", "content", "speech_text")

#: 单条反应台词最长留这么多字，避免一条长段落把见闻撑爆。
_MAX_LINE_CHARS = 200

#: 一条视频最多收这么多条反应。
_MAX_REACTIONS = 40


class HostWatchError(RuntimeError):
    """宿主 watch-together 调用失败。带上人话原因，方便写进日志和状态。"""


def _collect_text(payload: Any, out: list[str], depth: int = 0) -> None:
    """从任意嵌套结构里捞出反应台词。

    不写死时间轴的字段路径：项目里时间轴的形状会随版本改（``events`` 里放过
    ``text``，也放过 ``speech``），而这里只要"她说了什么"。所以按字段名认，
    不按位置认——多收几条重复的没关系，去重在后头。
    """
    if depth > 8:
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str) and key in _TEXT_KEYS:
                text = value.strip()
                if text:
                    out.append(text[:_MAX_LINE_CHARS])
            else:
                _collect_text(value, out, depth + 1)
    elif isinstance(payload, list):
        for item in payload:
            _collect_text(item, out, depth + 1)


class HostWatchService:
    """宿主 watch-together 的瘦客户端。所有方法都不抛异常给上层。"""

    def __init__(self, plugin: Any):
        self.plugin = plugin
        self._logger = plugin.logger
        #: 最近一次调用失败的原因（空串=最近一次是好的）。调用方据此判断
        #: 是"这次没货"（正常的 video=null）还是"宿主那边出毛病了"。
        self.last_error = ""
        self._pool: Any = None

    # ── 基础 ──

    def _base(self) -> str:
        settings = getattr(self.plugin, "_qq_settings", None) or {}
        base = str(settings.get("neko_host_api_base") or "").strip()
        if not base:
            base = os.environ.get("NEKO_HOST_API_BASE", "").strip()
        return (base or _DEFAULT_BASE).rstrip("/")

    def _request_sync(
        self, method: str, path: str, payload: dict[str, Any] | None, timeout: float,
    ) -> Any:
        """同步发一个请求。由 ``_request`` 丢进线程池跑。"""
        import urllib.error
        import urllib.request

        url = self._base() + path
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        # 显式空代理：宿主环境可能带系统代理变量，本机回环不该走代理。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=timeout) as response:  # noqa: S310 本地回环
                status = int(getattr(response, "status", 200) or 200)
                text = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001
                detail = ""
            raise HostWatchError(f"{method} {path} -> HTTP {exc.code}: {detail}") from None
        except TimeoutError:
            raise HostWatchError(f"{method} {path} 超时（>{timeout:.0f}s）") from None
        except OSError as exc:
            raise HostWatchError(f"{method} {path} 连不上: {exc}") from None
        if status >= 400:
            raise HostWatchError(f"{method} {path} -> HTTP {status}: {text[:200]}")
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError as exc:
            raise HostWatchError(f"{method} {path} 返回不是 JSON: {exc}") from None

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None, *, timeout: float,
    ) -> Any:
        # 同步发、不丢 asyncio 线程池。宿主这个插件进程里 asyncio 的线程池是坏的：
        # ``asyncio.to_thread`` / anyio 线程池 / run_coroutine_threadsafe 全都停在
        # "要个线程"那一步，协程永远醒不过来（实测四种写法，日志都停在发请求那一行）。
        #
        # 但直接内联同步调用又有另一个坑：prepare 一旦卡住（免费 TTS 排队时宿主
        # 会一直等到它自己的 1800 秒 deadline），urllib 的 socket 超时在这种同进程
        # 场景下不可靠。所以用自建线程池 + **调用方这边**的硬到期：work 跑不跑得动
        # 都不影响 result(timeout=) 到点返回，插件循环不会被拖死。
        self._logger.info(f"[HostWatch] → {method} {path} timeout={timeout:.0f}s")
        future = self._executor().submit(self._request_sync, method, path, payload, timeout)
        try:
            data = future.result(timeout=timeout + 5.0)
        except FutureTimeout:
            future.cancel()
            raise HostWatchError(f"{method} {path} 超时（>{timeout + 5.0:.0f}s）") from None
        self._logger.info(f"[HostWatch] ← {method} {path} ok")
        return data

    def _executor(self) -> Any:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="neko-host-watch")
        return self._pool

    def available(self) -> bool:
        return bool(self.plugin is not None)

    # ── discover ──

    async def discover(
        self, topic: str, exclude: Iterable[str] = (),
    ) -> dict[str, Any] | None:
        """让项目自己挑一条视频。挑不到返回 None（不是错误）。"""
        exclude_list = [str(item)[:32] for item in list(exclude)[:128] if str(item)]
        try:
            data = await self._request(
                "POST", "/api/watch-together/discover",
                {"topic": str(topic or "")[:200], "exclude": exclude_list},
                timeout=_DISCOVER_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            self._logger.warning(f"[HostWatch] discover 失败 topic={topic!r}: {exc}")
            return None
        self.last_error = ""
        if not isinstance(data, dict):
            return None
        video = data.get("video")
        if not isinstance(video, dict) or not video.get("bvid"):
            return None
        video = dict(video)
        video["topic"] = str(data.get("topic") or topic or "")
        return video

    # ── prepare ──

    async def prepare(self, url: str) -> str | None:
        """提交一次"真的看一遍"。返回 job id；提交不上返回 None。"""
        try:
            data = await self._request(
                "POST", "/api/watch-together/prepare",
                {"url": str(url), "source": "discovery"},
                timeout=60.0,
            )
        except Exception as exc:  # noqa: BLE001
            # 同一时刻只允许一个 prepare，撞上会 4xx——这是正常的排队语义。
            self._logger.info(f"[HostWatch] prepare 未受理: {exc}")
            return None
        if isinstance(data, dict):
            job = data.get("id") or data.get("job")
            if job:
                return str(job)
        return None

    async def preparation(self, job: str) -> dict[str, Any] | None:
        try:
            data = await self._request(
                "GET", f"/api/watch-together/preparation/{job}", None, timeout=30.0,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(f"[HostWatch] 查询准备进度失败 job={job}: {exc}")
            return None
        return data if isinstance(data, dict) else None

    async def wait(self, job: str, *, timeout: float = _DEFAULT_PREPARE_TIMEOUT) -> dict[str, Any] | None:
        """等到终态。超时返回最后一次看到的进度（带 timed_out 标记）。"""
        deadline = asyncio.get_running_loop().time() + timeout
        last: dict[str, Any] | None = None
        while asyncio.get_running_loop().time() < deadline:
            detail = await self.preparation(job)
            if detail is not None:
                last = detail
                status = str(detail.get("status") or "")
                if status in ("ready", "completed", "error", "cancelled"):
                    return detail
            await asyncio.sleep(_POLL_INTERVAL)
        if last is not None:
            last = dict(last)
            last["timed_out"] = True
        return last

    async def timeline(self, job: str, version: str | None = None) -> dict[str, Any] | None:
        try:
            data = await self._request(
                "GET", f"/api/watch-together/jobs/{job}/{version or job}", None, timeout=30.0,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.info(f"[HostWatch] 取时间轴失败 job={job}: {exc}")
            return None
        return data if isinstance(data, dict) else None

    # ── 组合：一条视频的"看过"结果 ──

    async def watch(self, url: str, *, timeout: float = _DEFAULT_PREPARE_TIMEOUT) -> dict[str, Any]:
        """看一条视频，返回 ``{ok, job, reactions, title, warnings, error}``。

        任何一步失败都返回 ``ok=False`` 并带上 ``error``——调用方据此降级，
        不要让它变成整轮刷 B 站失败的理由。
        """
        job = await self.prepare(url)
        if not job:
            return {"ok": False, "job": "", "reactions": [], "error": "宿主未受理 prepare"}
        detail = await self.wait(job, timeout=timeout)
        if detail is None:
            return {"ok": False, "job": job, "reactions": [], "error": "拿不到准备进度"}
        status = str(detail.get("status") or "")
        title = str(detail.get("title") or "")
        warnings = [str(item) for item in (detail.get("warnings") or [])]
        if status not in ("ready", "completed"):
            reason = str(detail.get("error") or status or "未知")
            timed_out = bool(detail.get("timed_out"))
            if timed_out:
                reason = f"超时（最后阶段 {detail.get('stage_key') or '?'}）"
            return {"ok": False, "job": job, "reactions": [], "title": title,
                    "warnings": warnings, "error": reason, "timed_out": timed_out}
        lines: list[str] = []
        _collect_text(detail.get("events") or [], lines)
        if not lines:
            # events 里已经带着她的台词时不必再取时间轴：那个路由按
            # (job, version) 索引，version 不等于 job id 时直接 404。
            timeline = await self.timeline(job)
            if timeline:
                _collect_text(timeline, lines)
        seen: set[str] = set()
        reactions: list[str] = []
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            reactions.append(line)
            if len(reactions) >= _MAX_REACTIONS:
                break
        return {
            "ok": True, "job": job, "reactions": reactions, "title": title,
            "warnings": warnings, "error": "",
            "duration": detail.get("duration") or 0,
            "bvid": str(detail.get("bvid") or ""),
        }