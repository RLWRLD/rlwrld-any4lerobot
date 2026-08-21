#!/usr/bin/env bash
# Pull a few real episodes for development. The full release is 114.28 TB, two
# episodes per embodiment is 9.6 GB, which is what the tests here need.
#
#   ./fetch_sample.sh tienyi 2
#
# Requires an AWS profile with read access to the foundry mirror. The `default`
# profile may be expired; `rlwrld` and `company-sso` both work.
set -euo pipefail

EMBODIMENT="${1:?usage: fetch_sample.sh <embodiment> [count]}"
COUNT="${2:-2}"
PROFILE="${AWS_PROFILE:-rlwrld}"
BUCKET="rlwrld-foundry-data"
DEST="${DEST:-./sample}"

# embodiment directory -> the repo slug that holds it. Franka lives in five.
case "$EMBODIMENT" in
  franka)         SLUGS=(franka-part-1) ;;
  franka_sim)     SLUGS=(franka-sim) ;;
  ur)             SLUGS=(ur5) ;;
  ur_dex)         SLUGS=(ur5-dex) ;;
  agilex)         SLUGS=(agilex) ;;
  agilex_mobile)  SLUGS=(agilex-mobile) ;;
  ark)            SLUGS=(ark) ;;
  ark_mobile)     SLUGS=(ark-mobile) ;;
  tienkung)       SLUGS=(tienkung) ;;
  tienkung_sim)   SLUGS=(tienkung-sim) ;;
  tienyi)         SLUGS=(tianyi) ;;
  tienyi_mobile)  SLUGS=(tianyi-mobile) ;;
  *) echo "unknown embodiment: $EMBODIMENT" >&2; exit 1 ;;
esac

for slug in "${SLUGS[@]}"; do
  prefix="external/robomind_2_0__robomind2-0-${slug}/2026-08-11/"
  echo "listing s3://$BUCKET/$prefix ..."
  aws s3 ls "s3://$BUCKET/$prefix" --recursive --profile "$PROFILE" \
    | awk '$4 ~ /\.hdf5$/ && $3 > 20000 {print $3, $4}' \
    | head -n "$COUNT" \
    | while read -r size key; do
        # keep the tree shape the converter discovers: data/<emb>/<task>/...
        rel="${key#"$prefix"}"
        out="$DEST/$EMBODIMENT/$rel"
        mkdir -p "$(dirname "$out")"
        echo "  $(( size / 1000000 )) MB  $rel"
        aws s3 cp "s3://$BUCKET/$key" "$out" --profile "$PROFILE" --only-show-errors
      done
  # the task's description file, where the release has one
  aws s3 cp "s3://$BUCKET/$prefix" "$DEST/$EMBODIMENT/" --recursive \
    --exclude "*" --include "*/zh_description.txt" --profile "$PROFILE" \
    --only-show-errors || true
done

echo "done. convert with:"
echo "  python robomind_v2_h5.py --src-paths $DEST/$EMBODIMENT --output-path ./out --debug"
