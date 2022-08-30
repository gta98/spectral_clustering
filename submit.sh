REPODIR=${REPODIR:-'.'}
SAVEDIR=${SAVEDIR:-'.'}

pushd .

ID1="REDACTED"
ID2="REDACTED"
FOLDERNAME=${ID1}_${ID2}_final
FILENAME=${FOLDERNAME}.zip

WORKDIR=/tmp/spkmeans/submission/${FOLDERNAME}
rm -rf $WORKDIR
mkdir -p $WORKDIR
mkdir $WORKDIR/generics
mkdir $WORKDIR/algorithms

cp -r $REPODIR/*.py $WORKDIR
cp -r $REPODIR/*.c $WORKDIR
cp -r $REPODIR/*.h $WORKDIR
cp -r $REPODIR/generics/*.c $WORKDIR/generics
cp -r $REPODIR/generics/*.h $WORKDIR/generics
cp -r $REPODIR/algorithms/*.c $WORKDIR/algorithms
cp -r $REPODIR/algorithms/*.h $WORKDIR/algorithms
cp -r $REPODIR/comp.sh $WORKDIR

cd $WORKDIR/..
rm $FOLDERNAME/test*.py
rm $FOLDERNAME/README.md
zip -r $FOLDERNAME.zip $FOLDERNAME
echo Folder is $FOLDERNAME in $WORKDIR
popd
cp $WORKDIR/../$FOLDERNAME.zip $SAVEDIR

echo "Saved submission in $SAVEDIR"
