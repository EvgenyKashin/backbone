## Setting AWS
Install docker https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1

Install nvidia docker (optional) https://github.com/NVIDIA/nvidia-docker 

sudo docker build -t backbone .

sudo docker run --rm -ti --ipc=host -p 8889:8889 -v `pwd`:/workspace backbone

jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port=8889

## Vast
ssh -p 16659 root@ssh4.vast.ai -L 8080:localhost:8080
scp -P 16659 -r backbone root@ssh4.vast.ai:/root

## Data
### Hymenoptera
wget https://download.pytorch.org/tutorial/hymenoptera_data.zip
unzip hymenoptera_data.zip 
mv hymenoptera_data data/
rm hymenoptera_data.zip 

### Cat dog
From https://www.kaggle.com/c/dogs-vs-cats/data

wget 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/3362/all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1558598123&Signature=dx6Qn2Oe6P%2BHPvP2MhExeVL7gxcbtC7zuqJMSm63Dd%2F5YqRBcNepRcvAkiWHWNdNJ6cL%2FdJi3MnEbBLan9dTTMAjlDJ9%2BZgn5b5if3ir0osaAt%2FBAfNleNSfSFZpH2inEdhj4%2FeBo%2Fz35ryVPbp5jk7EO%2BTDKHsWCTsj2ie9BgFP3cydZEew4iUwOLEdMiq%2FfKtPZnqODHaMibsknbfUpBGOhy2wd8HWGRfwWoZSCVZYAlJYYyRC5ssKaauePPZvefRAzhn6Xbro3xbNMZ6dx8GqFsH7QxMBk9USOvy6oDZH8bsyJ2b5n7ySh0owX7M0v8m3afyo1tvOpdRB4%2B83zg%3D%3D&response-content-disposition=attachment%3B+filename%3Ddogs-vs-cats.zip'  --no-check-certificate -O cat_dog.zip
unzip cat_dog.zip
unzip train.zip
rm cat_dog.zip
rm train.zip
rm test1.zip

### Tiny ImageNET
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
wget https://raw.githubusercontent.com/seshuad/IMagenet/master/tiny-imagenet-200/words.txt
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip 

EBS 0.1 * 200 * 43200  /(86400 * 30) =

rsync -av -e "ssh -p 18180" --exclude=".idea" --exclude=".git" --exclude="notebooks" backbone root@ssh5.vast.ai:/root
