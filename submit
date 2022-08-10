pushd .
SAVEDIR=${SAVEDIR:-'.'}

ID1="REDACTED"
ID2="REDACTED"
FOLDERNAME=${ID1}_${ID2}_final
FILENAME=${FOLDERNAME}.zip

WORKDIR=/tmp/spkmeans/submission/${FOLDERNAME}
rm -rf $WORKDIR
mkdir -p $WORKDIR
mkdir $WORKDIR/generics
mkdir $WORKDIR/algorithms

cp -r *.py $WORKDIR
cp -r *.c $WORKDIR
cp -r *.h $WORKDIR
cp -r generics/*.c $WORKDIR/generics
cp -r generics/*.h $WORKDIR/generics
cp -r algorithms/*.c $WORKDIR/algorithms
cp -r algorithms/*.h $WORKDIR/algorithms

cd $WORKDIR/..
zip -r $FOLDERNAME.zip $FOLDERNAME
popd
cd $SAVEDIR
cp $WORKDIR/../$FOLDERNAME.zip .
