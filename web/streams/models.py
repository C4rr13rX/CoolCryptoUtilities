from __future__ import annotations

from django.db import models


class MarketStream(models.Model):
    ts = models.FloatField()
    chain = models.CharField(max_length=64)
    symbol = models.CharField(max_length=64)
    price = models.FloatField()
    volume = models.FloatField()
    raw = models.JSONField()

    class Meta:
        db_table = "market_stream"
        managed = False
        ordering = ["-ts"]

    def as_dict(self) -> dict:
        return {
            "timestamp": self.ts,
            "chain": self.chain,
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "raw": self.raw,
        }
