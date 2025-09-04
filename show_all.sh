#!/usr/bin/env bash
# show_all.sh — tampilkan seluruh struktur dan isi file

cd ~/icame_project || { echo "Folder proyek tidak ditemukan"; exit 1; }

echo "=== Struktur Direktori ==="
if command -v tree >/dev/null 2>&1; then
  tree -a -L 2
else
  ls -R .
fi

echo
echo "=== Isi Semua File Teks ==="
find . -maxdepth 2 -type f | sort | while IFS= read -r file; do
  echo
  echo "----- FILE: $file -----"
  sed -n '1,500p' "$file"
  if [ "$(wc -l < "$file")" -gt 500 ]; then
    echo "[...isi terpotong setelah 500 baris...]"
  fi
done
