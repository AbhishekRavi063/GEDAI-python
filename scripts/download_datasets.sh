#!/usr/bin/env bash
# Download the EEG datasets for the gedai 0.5.0 rerun, straight into the folders
# MOABB and the benchmark loaders read from.
#
#   bash scripts/download_datasets.sh <target> [--dry-run] [--keep-zips]
#
#   targets:  shin     Shin2017A, 6 zips (~6.3 GB)
#             d27      Dataset 27, the 29 missing gel + water files (~5.7 GB)
#             weibo    Weibo2014, 3 zips (~4.4 GB)
#             bnci     BNCI2014_001: finish or repair the 18 files (~0.8 GB)
#             check    verify every dataset, download nothing
#             all      shin + d27 + weibo  (never touches BNCI)
#
# Safe to re-run: finished files are skipped and partial downloads resume.
# Zhou2016 is not here: MOABB fetches it itself (126 MB, BIDS layout).
# INRIA needs a Kaggle login, so it is printed as instructions only.

set -euo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MNE="${MNE_DATA:-$HOME/mne_data}"

BNCI_DIR="$MNE/MNE-bnci-data/~bci/database/001-2014"
SHIN_DIR="$MNE/MNE-eegfnirs-data"
WEIBO_DIR="$MNE/MNE-weibo-2014"
D27_DIR="$BENCH/datasets/dataset27"
INRIA_DIR="$BENCH/datasets/inria_bci_challenge/train"

LAMPX="https://lampx.tugraz.at/~bci/database"
SHIN_URL="http://doc.ml.tu-berlin.de/hBCI/EEG"
DATAVERSE="https://dataverse.harvard.edu/api/access/datafile"

DRY_RUN=0
KEEP_ZIPS=0
TARGET=""
for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY_RUN=1 ;;
    --keep-zips) KEEP_ZIPS=1 ;;
    -*)          echo "unknown option: $arg" >&2; exit 2 ;;
    *)           TARGET="$arg" ;;
  esac
done

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# Remote size in bytes, or empty if the server does not report one.
remote_size() {
  curl -sIL --max-time 30 "$1" | tr -d '\r' \
    | awk 'tolower($1)=="content-length:" {n=$2} END {print n}'
}

local_size() { stat -f%z "$1" 2>/dev/null || echo 0; }

# A file counts as done when it exists and, if the server reports a size, matches it.
is_complete() {
  local path="$1" url="$2" want
  [[ -s "$path" ]] || return 1
  [[ -f "$path.aria2" ]] && return 1          # aria2c still has it open
  want="$(remote_size "$url")"
  [[ -z "$want" || "$(local_size "$path")" == "$want" ]]
}

# Download a batch. Each line of the list file is: <dir>\t<filename>\t<url>
fetch_batch() {
  local list="$1" n
  n="$(grep -c . "$list" || true)"
  [[ "$n" -gt 0 ]] || { log "nothing to download"; return 0; }

  if (( DRY_RUN )); then
    while IFS=$'\t' read -r dir name url; do
      printf '  would fetch %-16s -> %s\n' "$name" "$dir"
    done < "$list"
    return 0
  fi

  while IFS=$'\t' read -r dir name _; do mkdir -p "$dir"; done < "$list"

  if command -v aria2c >/dev/null; then
    log "aria2c: $n file(s), 16 connections each, 4 files at a time"
    local input; input="$(mktemp)"
    while IFS=$'\t' read -r dir name url; do
      printf '%s\n  dir=%s\n  out=%s\n' "$url" "$dir" "$name"
    done < "$list" > "$input"
    aria2c -i "$input" -j 4 -x 16 -s 16 -k 1M -c \
      --auto-file-renaming=false --allow-overwrite=true \
      --file-allocation=none --max-tries=10 --retry-wait=5 \
      --console-log-level=warn --summary-interval=30
    rm -f "$input"
  else
    log "curl: $n file(s), 4 at a time (brew install aria2 for faster downloads)"
    tr '\t' '\n' < "$list" | xargs -n 3 -P 4 sh -c \
      'curl -fL -C - --retry 10 --retry-delay 5 --silent --show-error -o "$0/$1" "$2" \
         && echo "  done $1"'
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# BNCI2014_001
# ─────────────────────────────────────────────────────────────────────────────
do_bnci() {
  log "BNCI2014_001 -> $BNCI_DIR"
  local list; list="$(mktemp)"
  for s in $(seq 1 9); do
    for r in T E; do
      local name; name="$(printf 'A%02d%s.mat' "$s" "$r")"
      local url="$LAMPX/001-2014/$name"
      if is_complete "$BNCI_DIR/$name" "$url"; then
        printf '  ok  %s\n' "$name"
      else
        printf '%s\t%s\t%s\n' "$BNCI_DIR" "$name" "$url" >> "$list"
      fi
    done
  done
  fetch_batch "$list"; rm -f "$list"
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset 27: gel G01-G15, water V01-V15, dry H01-H15
# ─────────────────────────────────────────────────────────────────────────────
do_d27() {
  log "Dataset 27 -> $D27_DIR"
  local list; list="$(mktemp)"
  local have=0
  for prefix in G V H; do
    for i in $(seq 1 15); do
      local name; name="$(printf '%s%02d.mat' "$prefix" "$i")"
      local url="$LAMPX/001-2020/$name"
      if is_complete "$D27_DIR/$name" "$url"; then
        have=$((have + 1))
      else
        printf '%s\t%s\t%s\n' "$D27_DIR" "$name" "$url" >> "$list"
      fi
    done
  done
  log "  $have of 45 already complete"
  fetch_batch "$list"; rm -f "$list"
}

# ─────────────────────────────────────────────────────────────────────────────
# Shin2017A: six zips, each extracted into EEG/ as "subject NN/with occular artifact"
# ─────────────────────────────────────────────────────────────────────────────
shin_subject_ok() {
  [[ -s "$SHIN_DIR/EEG/$(printf 'subject %02d' "$1")/with occular artifact/cnt.mat" ]]
}

do_shin() {
  log "Shin2017A -> $SHIN_DIR/EEG"
  local ranges=(01-05 06-10 11-15 16-20 21-25 26-29)
  local list; list="$(mktemp)"
  local todo=()

  for rng in "${ranges[@]}"; do
    local lo=$((10#${rng%-*})) hi=$((10#${rng#*-})) missing=0
    for s in $(seq "$lo" "$hi"); do shin_subject_ok "$s" || missing=1; done
    if (( missing )); then
      todo+=("$rng")
      local zip="EEG_$rng.zip" url="$SHIN_URL/EEG_$rng.zip"
      is_complete "$SHIN_DIR/$zip" "$url" \
        || printf '%s\t%s\t%s\n' "$SHIN_DIR" "$zip" "$url" >> "$list"
    else
      printf '  ok  subjects %s\n' "$rng"
    fi
  done

  fetch_batch "$list"; rm -f "$list"
  (( DRY_RUN )) && return 0

  for rng in "${todo[@]}"; do
    local zip="$SHIN_DIR/EEG_$rng.zip"
    log "  extracting EEG_$rng.zip"
    unzip -tq "$zip" >/dev/null || fail "$zip is corrupt; delete it and re-run"
    mkdir -p "$SHIN_DIR/EEG"
    unzip -oq "$zip" -d "$SHIN_DIR/EEG"
    local lo=$((10#${rng%-*})) hi=$((10#${rng#*-}))
    for s in $(seq "$lo" "$hi"); do
      shin_subject_ok "$s" || fail "subject $s not found after extracting $zip. \
Top level of the zip: $(unzip -Z1 "$zip" | cut -d/ -f1 | sort -u | head -3 | tr '\n' ' ')"
    done
    (( KEEP_ZIPS )) || rm -f "$zip"
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# Weibo2014: three zips, files renamed to subject_N.mat exactly as MOABB does
# ─────────────────────────────────────────────────────────────────────────────
do_weibo() {
  log "Weibo2014 -> $WEIBO_DIR"
  # zip-name | dataverse id | subjects | filename prefixes, in the same order
  local specs=(
    "data1.zip|2499178|1 2 3 4|cl cyy kyf lnn"
    "data2_2.zip|2499182|5 6 7|ls ry wcf"
    "data3.zip|2499179|8 9 10|wx yyx zd"
  )
  local list; list="$(mktemp)"
  local todo=()

  for spec in "${specs[@]}"; do
    IFS='|' read -r zip id subs _ <<< "$spec"
    local missing=0
    for s in $subs; do [[ -s "$WEIBO_DIR/subject_$s.mat" ]] || missing=1; done
    if (( missing )); then
      todo+=("$spec")
      [[ -s "$WEIBO_DIR/$zip" && ! -f "$WEIBO_DIR/$zip.aria2" ]] \
        || printf '%s\t%s\t%s\n' "$WEIBO_DIR" "$zip" "$DATAVERSE/$id" >> "$list"
    else
      printf '  ok  subjects %s\n' "$subs"
    fi
  done

  fetch_batch "$list"; rm -f "$list"
  (( DRY_RUN )) && return 0

  for spec in "${todo[@]}"; do
    IFS='|' read -r zip id subs prefixes <<< "$spec"
    local zpath="$WEIBO_DIR/$zip" tmp="$WEIBO_DIR/.unzip_${zip%.zip}"
    log "  extracting $zip"
    unzip -tq "$zpath" >/dev/null || fail "$zpath is corrupt; delete it and re-run"
    rm -rf "$tmp"; mkdir -p "$tmp"
    unzip -oq "$zpath" -d "$tmp"

    local subs_a=($subs) pref_a=($prefixes)
    for k in "${!subs_a[@]}"; do
      local s="${subs_a[$k]}" p="${pref_a[$k]}"
      local matches=()
      while IFS= read -r f; do matches+=("$f"); done < <(
        find "$tmp" -type f -name "${p}*" ! -name '._*')
      [[ ${#matches[@]} -eq 1 ]] || fail "expected one file starting '$p' in $zip, \
found ${#matches[@]}: $(find "$tmp" -type f ! -name '._*' -exec basename {} \; | tr '\n' ' ')"
      mv -f "${matches[0]}" "$WEIBO_DIR/subject_$s.mat"
      printf '  %-6s -> subject_%s.mat\n' "$p" "$s"
    done
    rm -rf "$tmp"
    (( KEEP_ZIPS )) || rm -f "$zpath"
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# Verify everything, download nothing
# ─────────────────────────────────────────────────────────────────────────────
do_check() {
  local n

  n=0; for s in $(seq 1 9); do for r in T E; do
    [[ -s "$BNCI_DIR/$(printf 'A%02d%s.mat' "$s" "$r")" ]] && n=$((n + 1)); done; done
  printf '%-14s %2d / 18   %s\n' "BNCI2014_001" "$n" "$BNCI_DIR"

  n=0; for s in $(seq 1 29); do shin_subject_ok "$s" && n=$((n + 1)); done
  printf '%-14s %2d / 29   %s\n' "Shin2017A" "$n" "$SHIN_DIR/EEG"

  n=0; for s in $(seq 1 10); do [[ -s "$WEIBO_DIR/subject_$s.mat" ]] && n=$((n + 1)); done
  printf '%-14s %2d / 10   %s\n' "Weibo2014" "$n" "$WEIBO_DIR"

  for prefix in G V H; do
    n=0; for i in $(seq 1 15); do
      [[ -s "$D27_DIR/$(printf '%s%02d.mat' "$prefix" "$i")" ]] && n=$((n + 1)); done
    printf '%-14s %2d / 15   %s\n' "Dataset27 $prefix" "$n" "$D27_DIR"
  done

  n="$(ls "$INRIA_DIR"/Data_S*_Sess*.csv 2>/dev/null | wc -l | tr -d ' ')"
  printf '%-14s %2d / 80   %s\n' "INRIA" "$n" "$INRIA_DIR"
  printf '%-14s  auto      fetched by MOABB on first use\n' "Zhou2016"
}

inria_note() {
  cat <<EOF

INRIA is not downloaded by this script; it needs a Kaggle login.
  1. Sign in and accept the rules: https://www.kaggle.com/c/inria-bci-challenge/rules
  2. Download train.zip:          https://www.kaggle.com/c/inria-bci-challenge/data
  3. unzip -o train.zip -d "$(dirname "$INRIA_DIR")"
EOF
}

case "$TARGET" in
  bnci)  do_bnci ;;
  d27)   do_d27 ;;
  shin)  do_shin ;;
  weibo) do_weibo ;;
  check) do_check; exit 0 ;;
  all)   do_d27; do_shin; do_weibo ;;          # BNCI excluded: downloaded manually
  *)     sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

(( DRY_RUN )) && { log "dry run: nothing was downloaded"; exit 0; }
log "finished $TARGET"
echo
do_check
[[ "$TARGET" == "all" ]] && inria_note
exit 0
