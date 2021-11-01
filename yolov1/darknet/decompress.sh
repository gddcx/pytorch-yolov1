#ILSVRC2012
#  |-ILSVRC2012.img_train.tar
#  |-ILSVRC2012.img_val.tar
#  |-train
#  |-val

cd ../../dataset/ILSVRC2012
tar -xvf ILSVRC2012.img_train.tar -C train
tar -xvf ILSVRC2012.img_val.tar -C val

cd train
for name in `ls | xargs -i echo {} | cut -c 1-9`
do
  mkdir $name
  tar -xvf $name'.tar' -C $name
done


