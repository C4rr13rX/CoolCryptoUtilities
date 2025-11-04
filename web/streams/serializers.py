from __future__ import annotations

from rest_framework import serializers

from .models import MarketStream


class MarketStreamSerializer(serializers.ModelSerializer):
    class Meta:
        model = MarketStream
        fields = ["ts", "chain", "symbol", "price", "volume", "raw"]
