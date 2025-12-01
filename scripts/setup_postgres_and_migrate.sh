#!/usr/bin/env bash
# Installs PostgreSQL (Ubuntu/WSL), provisions a default database/user,
# and migrates the current SQLite data into PostgreSQL.
#
# Usage:
#   sudo bash scripts/setup_postgres_and_migrate.sh

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "❌  Please run this script with sudo/root privileges."
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_USER="$(stat -c '%U' "$PROJECT_ROOT")"
ROLE_NAME="coolcrypto"
DB_NAME="coolcrypto"
ROLE_PASSWORD=${POSTGRES_BOOTSTRAP_PASSWORD:-coolcrypto_password}
PG_HOST="127.0.0.1"
PG_PORT="5432"
ENV_FILE="$PROJECT_ROOT/.env.postgres"

echo "📦 Installing PostgreSQL server packages…"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y postgresql postgresql-contrib

echo "🚀 Ensuring PostgreSQL service is running…"
service postgresql start

echo "🗄️  Creating database role and schema (role: $ROLE_NAME, db: $DB_NAME)…"
sudo -u postgres psql <<SQL
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$ROLE_NAME') THEN
        CREATE ROLE $ROLE_NAME LOGIN PASSWORD '$ROLE_PASSWORD';
    ELSE
        ALTER ROLE $ROLE_NAME WITH LOGIN PASSWORD '$ROLE_PASSWORD';
    END IF;
END
\$\$;

DO \$\$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = '$DB_NAME') THEN
        CREATE DATABASE $DB_NAME OWNER $ROLE_NAME;
    ELSE
        ALTER DATABASE $DB_NAME OWNER TO $ROLE_NAME;
    END IF;
END
\$\$;
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $ROLE_NAME;
SQL

echo "📝 Writing connection defaults to $ENV_FILE (update as needed)…"
cat > "$ENV_FILE" <<EOF
DJANGO_DB_VENDOR=postgres
POSTGRES_HOST=$PG_HOST
POSTGRES_PORT=$PG_PORT
POSTGRES_DB=$DB_NAME
POSTGRES_USER=$ROLE_NAME
POSTGRES_PASSWORD=$ROLE_PASSWORD
POSTGRES_SSLMODE=disable
EOF

echo "♻️  Migrating existing SQLite data into PostgreSQL…"
sudo -u "$PROJECT_USER" bash -c "
  source '$PROJECT_ROOT/bin/activate'
  export DJANGO_DB_VENDOR=postgres
  export POSTGRES_HOST=$PG_HOST
  export POSTGRES_PORT=$PG_PORT
  export POSTGRES_DB=$DB_NAME
  export POSTGRES_USER=$ROLE_NAME
  export POSTGRES_PASSWORD=$ROLE_PASSWORD
  export POSTGRES_SSLMODE=disable
  python3 '$PROJECT_ROOT/scripts/sqlite_to_postgres.py'
"

cat <<'NOTE'
✅ PostgreSQL is ready and data has been migrated.

Next steps:
 1. Review .env.postgres (and copy values into your main .env/.bashrc).
 2. Update POSTGRES_PASSWORD to a strong secret (`sudo -u postgres psql` → `ALTER ROLE …`).
 3. Reseat Django with the new env:
        source ./bin/activate
        export $(cat .env.postgres | xargs)
        python3 web/manage.py reseat --guardian-off
NOTE
