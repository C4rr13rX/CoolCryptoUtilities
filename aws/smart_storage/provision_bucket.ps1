# Provision a SmartStorageService bucket in YOUR AWS account.
#
#   ./provision_bucket.ps1 -Bucket my-market-data-<accountid> -Profile myprofile
#
# Creates the bucket, blocks all public access, enables AES256 encryption with
# a bucket key, and applies the tiering lifecycle. Safe to re-run: every step
# is idempotent, so this doubles as "bring an existing bucket up to spec".
#
# No account id, profile name or bucket name is baked in. Pass your own.
param(
    [Parameter(Mandatory = $true)][string]$Bucket,
    [Parameter(Mandatory = $true)][string]$Profile,
    [string]$Region = "us-east-1"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Output "account:"
aws sts get-caller-identity --profile $Profile --output text --query Account

# us-east-1 is the one region where CreateBucket must NOT carry a location
# constraint; every other region requires it.
Write-Output "creating bucket $Bucket in $Region ..."
if ($Region -eq "us-east-1") {
    aws s3api create-bucket --bucket $Bucket --region $Region --profile $Profile 2>$null
} else {
    aws s3api create-bucket --bucket $Bucket --region $Region `
        --create-bucket-configuration "LocationConstraint=$Region" --profile $Profile 2>$null
}

Write-Output "blocking public access ..."
aws s3api put-public-access-block --bucket $Bucket `
    --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" `
    --profile $Profile

# Market data is immutable once written, so object versioning would only ever
# double the bill for copies nothing reads. Deliberately left off.
Write-Output "enabling default encryption ..."
aws s3api put-bucket-encryption --bucket $Bucket `
    --server-side-encryption-configuration '{\"Rules\":[{\"ApplyServerSideEncryptionByDefault\":{\"SSEAlgorithm\":\"AES256\"},\"BucketKeyEnabled\":true}]}' `
    --profile $Profile

Write-Output "applying lifecycle (IA at 30d, Glacier-IR at 120d) ..."
$lifecycle = Join-Path $here "bucket_lifecycle.json"
aws s3api put-bucket-lifecycle-configuration --bucket $Bucket `
    --lifecycle-configuration "file://$lifecycle" --profile $Profile

Write-Output ""
Write-Output "done. add to your .env (which is gitignored):"
Write-Output "  MARKET_DATA_S3_ENABLED=1"
Write-Output "  MARKET_DATA_S3_BUCKET=$Bucket"
Write-Output "  MARKET_DATA_S3_PROFILE=$Profile"
Write-Output "  MARKET_DATA_S3_REGION=$Region"
