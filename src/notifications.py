from concurrent.futures import ThreadPoolExecutor
import requests
from threading import Lock


class NotificationService:
    _instance = None
    _lock = Lock()

    def __new__(cls, url=None, max_workers=1, enabled=True):
        """
        Singleton implementation for NotificationService.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NotificationService, cls).__new__(cls)
                cls._instance._init(url, max_workers, enabled)
        return cls._instance

    def _init(self, url, max_workers, enabled=True):
        """
        Initializes the NotificationService instance.
        """
        self.url = url
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.enabled = enabled

    def post(self, payload):
        """
        Sends a JSON payload to the configured URL asynchronously.
        """
        if not self.enabled:
            return

        if not self.url:
            raise ValueError("NotificationService URL is not configured.")

        def send_request():
            try:
                # fire and hopefully forget
                requests.post(self.url, json=payload, timeout=10)
            except Exception as e:
                print(f"warn: notification failed: {e}")

        self.executor.submit(send_request)

    @classmethod
    def configure(cls, url, max_workers=1, enabled=True):
        """
        Configures the singleton instance with a URL and max workers.
        """
        instance = cls(url, max_workers, enabled)
        return instance
