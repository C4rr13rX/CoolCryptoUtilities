#!/usr/bin/env bash
# Enforce shared ownership between root and adam for the CoolCryptoUtilities repo.
set -euo pipefail

ROOT_DIR="/home/adam/CoolCryptoUtilities"
GROUP="rootadam"
PRIMARY_USER="adam"

# Ensure group and memberships exist (idempotent)
if ! getent group "${GROUP}" >/dev/null; then
  groupadd -f "${GROUP}"
fi
usermod -a -G "${GROUP}" "${PRIMARY_USER}"
usermod -a -G "${GROUP}" root

# Baseline: fix ownership/permissions and defaults so both users can write.
chown -R root:"${GROUP}" "${ROOT_DIR}"
chmod -R g+rwX "${ROOT_DIR}"
find "${ROOT_DIR}" -type d -exec chmod g+s {} +
setfacl -R -m g:"${GROUP}":rwx "${ROOT_DIR}"
setfacl -dR -m g:"${GROUP}":rwx "${ROOT_DIR}"
setfacl -R -m u:"${PRIMARY_USER}":rwx "${ROOT_DIR}"
setfacl -dR -m u:"${PRIMARY_USER}":rwx "${ROOT_DIR}"

# Continuous watch: whenever something changes, fix the touched path.
command -v inotifywait >/dev/null 2>&1 || { echo "inotifywait is required"; exit 1; }
inotifywait -m -r -e create -e moved_to -e close_write -e attrib "${ROOT_DIR}" |
while read -r directory events filename; do
  target="${directory%/}/${filename}"
  [ -e "${target}" ] || continue

  chown root:"${GROUP}" "${target}" || true
  chmod g+rwX "${target}" || true
  setfacl -m u:"${PRIMARY_USER}":rwx "${target}" || true
  if [ -d "${target}" ]; then
    chmod g+s "${target}" || true
    setfacl -d -m g:"${GROUP}":rwx "${target}" || true
    setfacl -d -m u:"${PRIMARY_USER}":rwx "${target}" || true
  fi
done
