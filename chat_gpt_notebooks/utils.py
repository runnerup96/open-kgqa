"""
Модуль для работы с API запросами с поддержкой rate limiting и retry логики.
"""
import re
import asyncio
import traceback
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import defaultdict

from aiohttp import ClientSession, TCPConnector, ClientTimeout
from tenacity import retry, stop_after_attempt, wait_random_exponential
from time import sleep
from logging_module import setup_logger

logger = setup_logger('ClientLogger')

# Константы
DEFAULT_TIMEOUT = ClientTimeout(total=30, connect=10)
DEFAULT_CONNECTOR = TCPConnector(
    limit=100,
    ttl_dns_cache=300,
    force_close=False,
    enable_cleanup_closed=True
)


class RateLimiter:
    """Класс для управления rate limiting."""

    def __init__(self, request_limit=10, token_limit=1000):
        self.limits = {}
        self.usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        self.thresholds = {
            'requests': request_limit,
            'tokens': token_limit
        }

    async def check_and_wait(self):
        """Проверяет лимиты и ждет если необходимо."""
        if not self.limits:
            return

        for limit_type in ['requests', 'tokens']:
            remaining = self.limits.get('remaining', {}).get(limit_type, float('inf'))
            threshold = self.thresholds.get(limit_type)

            if remaining < threshold:
                reset_time = datetime.fromisoformat(self.limits['reset'][limit_type])
                wait_seconds = max(0, (reset_time - datetime.now()).total_seconds())
                if wait_seconds > 0:
                    logger.info(f"Достигнут лимит {limit_type}. Ожидание {wait_seconds:.2f} секунд")
                    await asyncio.sleep(wait_seconds)


class APIClient:
    """Класс для выполнения API запросов."""

    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.session = ClientSession(
            connector=DEFAULT_CONNECTOR,
            timeout=DEFAULT_TIMEOUT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "assistants=v2",
            }
        )
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter()
        self.semaphore = asyncio.Semaphore(10)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(self, url: str, body: Dict) -> Optional[Dict]:
        """Выполняет единичный API запрос."""
        response = None
        try:
            async with self.semaphore:
                async with self.session.post(url=url, json=body) as response:
                    content = await response.json()

                    if error := content.get('error'):
                        log_message = (
                            f"\n{'=' * 80}\n"
                            f"🚨 OpenAI Error Occured During API Request Call\n"
                            f"{'-' * 80}\n"
                            f"Code       : {error.get('code', 'N/A')}\n"
                            f"Message    : {error.get('message', 'N/A')}\n"
                            f"Response   : {response.status} ({response.reason})\n"
                            f"{'=' * 80}\n"
                        )
                        logger.error(log_message)
                        return None

                    self.rate_limiter.limits = self._parse_rate_limits(response)
                    return content

        except Exception as e:
            log_message = (
                f"\n{'=' * 80}\n"
                f"🚨 A General Exception Occurred During API Request Call\n"
                f"{'-' * 80}\n"
                f"Type       : {type(e).__name__}\n"
                f"Message    : {str(e)}\n"
                f"Response   : {response.status if response else 'N/A'} ({response.reason if response else 'N/A'})\n"
                f"Traceback  :\n{traceback.format_exc()}\n"
                f"{'=' * 80}\n"
            )
            logger.error(log_message)
            return None

    @retry(wait=wait_random_exponential(min=1, max=300), stop=stop_after_attempt(3), reraise=False)
    async def make_request_with_retry(self, url: str, body: Dict) -> Optional[Dict]:
        return await self.make_request(url=url, body=body)

    async def process_tasks(self, tasks: List[Dict], verbose: bool = False) -> List[Dict]:
        """Обрабатывает список задач с поддержкой повторных попыток."""
        failed_tasks = []
        completed_tasks = {}
        retry_counts = defaultdict(int)

        pbar = tqdm(total=len(tasks), position=0, leave=True, desc='Processing tasks')

        def update_status():
            remaining_requests = self.rate_limiter.limits.get('remaining', {}).get('requests', 'N/A')
            remaining_tokens = self.rate_limiter.limits.get('remaining', {}).get('tokens', 'N/A')
            pbar.set_description(
                f"Лимиты: {remaining_requests} запросов, {remaining_tokens} токенов | "
                f"Токены: {self.rate_limiter.usage['total_tokens']}"
            )
            if type(remaining_tokens) == int:
                if int(remaining_tokens) < 5000:
                    sleep(100)

        for task in tasks:
            task_id = task.get('task_id')

            await self.rate_limiter.check_and_wait()

            if not self.rate_limiter.limits:
                result = await self.make_request_with_retry(task['url'], task['body'])
            else:
                result = await self.make_request_with_retry(task['url'], task['body'])
                # result = await self.make_request(task['url'], task['body'])

            not_finished = any(
                [it['finish_reason'] not in ['stop', 'length'] for it in result.get('choices', [])]) if result else None

            if not result or not_finished:
                retry_counts[task_id] += 1
                if retry_counts[task_id] < self.max_retries:
                    logger.info(
                        f"Retrying task {task_id}. (Reason: {not_finished}) ({retry_counts[task_id]}/{self.max_retries})")
                    tasks.append(task)  # Добавляем задачу в конец очереди
                else:
                    logger.warning(f"Task {task_id} failed after {self.max_retries} attempts")
                    if verbose: print(f"🚫 Giving up on task {task_id} after {self.max_retries} attempts")
                    failed_tasks.append(task)
                    pbar.update(1)
            else:
                if verbose: print(f"✅ Final success for {task_id}")

                usage = result.get('usage', {})
                self.rate_limiter.usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
                self.rate_limiter.usage['completion_tokens'] += usage.get('completion_tokens', 0)
                self.rate_limiter.usage['total_tokens'] += usage.get('total_tokens', 0)

                completed_tasks[task_id] = [res['message'] for res in result['choices']]
                pbar.update(1)

            update_status()

        return completed_tasks, failed_tasks

    @staticmethod
    def _parse_to_timedelta(time_str):
        # Regular expression to extract all components (e.g., 6m0s -> [('6', 'm'), ('0', 's')])
        matches = re.findall(r'([\d.]+)([a-zA-Z]+)', time_str.strip())
        if not matches:
            raise ValueError(f"Invalid time format: {time_str}")

        # Initialize timedelta
        total_delta = timedelta()

        # Process each component
        for value, unit in matches:
            value = float(value)
            unit = unit.lower()

            if unit == 's':  # Seconds
                total_delta += timedelta(seconds=value)
            elif unit == 'ms':  # Milliseconds
                total_delta += timedelta(milliseconds=value)
            elif unit == 'm':  # Minutes
                total_delta += timedelta(minutes=value)
            elif unit == 'h':  # Hours
                total_delta += timedelta(hours=value)
            elif unit == 'd':  # Days
                total_delta += timedelta(days=value)
            else:
                return timedelta(seconds=0)
        return total_delta

    def _parse_rate_limits(self, response) -> Dict:
        """Парсит заголовки rate limiting из ответа."""
        try:
            return {
                'reset': {
                    key: (datetime.now() + self._parse_to_timedelta(
                        response.headers.get(f"x-ratelimit-reset-{key}"))).isoformat()
                    for key in ["requests", "tokens"]
                },
                'remaining': {
                    key: int(response.headers.get(f"x-ratelimit-remaining-{key}", float('inf')))
                    for key in ["requests", "tokens"]
                }
            }
        except Exception as e:
            logger.warning(f"Failed to parse rate limits: {e}")
            return {}
