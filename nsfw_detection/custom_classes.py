from typing import List


class DetectionResult:
    def __init__(self, url: str, score: float = 0.0, message: str = ""):
        self.url = url
        self.score = score
        self.message = message

    def __str__(self):
        return f"URL: {self.url}, Score: {self.score}, Message: {self.message}"

    def to_dict(self) -> dict:
        return (
            {
                "url": self.url,
                "score": self.score,
            }
            if not self.message
            else {
                "url": self.url,
                "message": self.message,
            }
        )


class NSFWDetectionResult:
    def __init__(self, nsfw: List[DetectionResult]):
        self.nsfw = nsfw

    def __str__(self):
        return f"NSFW: {self.nsfw}"

    def to_dict(self) -> dict:
        return {
            "nsfw": [e.to_dict() for e in self.nsfw] if self.nsfw else [],
        }
