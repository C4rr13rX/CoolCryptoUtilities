from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("branddozer", "0008_research_papers"),
    ]

    operations = [
        migrations.AddField(
            model_name="researchsource",
            name="first_party",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="researchsource",
            name="provenance_detail",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="researchsource",
            name="provenance_status",
            field=models.CharField(
                choices=[
                    ("verified", "Verified"),
                    ("corroborated", "Corroborated"),
                    ("unverified", "Unverified"),
                    ("disputed", "Disputed"),
                ],
                default="unverified",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="researchsource",
            name="source_class",
            field=models.CharField(
                choices=[
                    ("corporate_primary", "Corporate Primary"),
                    ("government_record", "Government Record"),
                    ("court_record", "Court Record"),
                    ("leaked_primary", "Leaked Primary"),
                    ("archival_copy", "Archival Copy"),
                    ("peer_reviewed", "Peer Reviewed"),
                    ("journalism", "Journalism"),
                    ("advocacy", "Advocacy"),
                    ("other", "Other"),
                ],
                default="other",
                max_length=32,
            ),
        ),
    ]
