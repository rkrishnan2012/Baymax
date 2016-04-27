# Remove spaces from files in subdirectory of images
for f in Filtered/palsy/*; do mv "$f" "${f// /_}"; done
for f in Filtered/regular/*; do mv "$f" "${f// /_}"; done

mkdir -p Sorted/palsy
mkdir -p Sorted/regular

cd Filtered
find palsy -type f |
gshuf |  # shuffle the input lines, i.e. apply a random permutation
nl -n rz |  # add line numbers 000001, …
while read -r number name; do
  ext=${name##*/}  # try to retain the file name extension
  case $ext in
    *.*) ext=.${ext##*.};;
    *) ext=;;
  esac
  mv "$name" "../Sorted/${name%/*}/$number$ext"
done

find regular -type f |
gshuf |  # shuffle the input lines, i.e. apply a random permutation
nl -n rz |  # add line numbers 000001, …
while read -r number name; do
  ext=${name##*/}  # try to retain the file name extension
  case $ext in
    *.*) ext=.${ext##*.};;
    *) ext=;;
  esac
  mv "$name" "../Sorted/${name%/*}/$number$ext"
done

cd ..


mkdir -p Sorted/train/palsy
mkdir -p Sorted/train/regular
mkdir -p Sorted/cv/palsy
mkdir -p Sorted/cv/regular
mkdir -p Sorted/test/palsy
mkdir -p Sorted/test/regular


totalFiles=$(ls -1 Sorted/palsy | wc -l)
train=$((totalFiles * 3 / 5))
test=$((totalFiles * 1 / 5))
cv=$((totalFiles * 1 / 5))

for file in $(ls -p Sorted/palsy | grep -v / | tail -$train)
do
mv Sorted/palsy/$file Sorted/train/palsy/
done

for file in $(ls -p Sorted/palsy | grep -v / | tail -$test)
do
mv Sorted/palsy/$file Sorted/test/palsy/
done

for file in $(ls -p Sorted/palsy | grep -v / | tail -$cv)
do
mv Sorted/palsy/$file Sorted/cv/palsy/
done

totalFiles=$(ls -1 Sorted/regular | wc -l)
train=$((totalFiles * 3 / 5))
test=$((totalFiles * 1 / 5))
cv=$((totalFiles * 1 / 5))

for file in $(ls -p Sorted/regular | grep -v / | tail -$train)
do
mv Sorted/regular/$file Sorted/train/regular/
done

for file in $(ls -p Sorted/regular | grep -v / | tail -$test)
do
mv Sorted/regular/$file Sorted/test/regular/
done

for file in $(ls -p Sorted/regular | grep -v / | tail -$cv)
do
mv Sorted/regular/$file Sorted/cv/regular/
done

rm -rf Sorted/regular
rm -rf Sorted/palsy