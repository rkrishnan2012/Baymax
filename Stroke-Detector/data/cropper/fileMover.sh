# Remove spaces from files in subdirectory of images
for f in filtered-images/palsy/*; do mv "$f" "${f// /_}"; done
for f in filtered-images/regular/*; do mv "$f" "${f// /_}"; done

mkdir -p sorted/palsy
mkdir -p sorted/regular

cd filtered-images
find palsy -type f |
gshuf |  # shuffle the input lines, i.e. apply a random permutation
nl -n rz |  # add line numbers 000001, …
while read -r number name; do
  ext=${name##*/}  # try to retain the file name extension
  case $ext in
    *.*) ext=.${ext##*.};;
    *) ext=;;
  esac
  mv "$name" "../sorted/${name%/*}/$number$ext"
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
  mv "$name" "../sorted/${name%/*}/$number$ext"
done

cd ..


mkdir -p sorted/train/palsy
mkdir -p sorted/train/regular
mkdir -p sorted/cv/palsy
mkdir -p sorted/cv/regular
mkdir -p sorted/test/palsy
mkdir -p sorted/test/regular


totalFiles=$(ls -1 sorted/palsy | wc -l)
train=$((totalFiles * 3 / 5))
test=$((totalFiles * 1 / 5))
cv=$((totalFiles * 1 / 5))

for file in $(ls -p sorted/palsy | grep -v / | tail -$train)
do
mv sorted/palsy/$file sorted/train/palsy/
done

for file in $(ls -p sorted/palsy | grep -v / | tail -$test)
do
mv sorted/palsy/$file sorted/test/palsy/
done

for file in $(ls -p sorted/palsy | grep -v / | tail -$cv)
do
mv sorted/palsy/$file sorted/cv/palsy/
done

totalFiles=$(ls -1 sorted/regular | wc -l)
train=$((totalFiles * 3 / 5))
test=$((totalFiles * 1 / 5))
cv=$((totalFiles * 1 / 5))

for file in $(ls -p sorted/regular | grep -v / | tail -$train)
do
mv sorted/regular/$file sorted/train/regular/
done

for file in $(ls -p sorted/regular | grep -v / | tail -$test)
do
mv sorted/regular/$file sorted/test/regular/
done

for file in $(ls -p sorted/regular | grep -v / | tail -$cv)
do
mv sorted/regular/$file sorted/cv/regular/
done

rm -rf sorted/regular
rm -rf sorted/palsy