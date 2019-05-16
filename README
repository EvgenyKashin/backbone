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
wget 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/3362/all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1558208774&Signature=OIneBgu8n7%2B1worGORzcY28SQvKVkmvq%2Fr4sjssF0yWMTVi%2BG1ofcrzd5HPmt0ZLrF8pyjZCglYjSbOsAah1nSWtHYbCD97wfFzrm%2BTMb70apJ%2BS1KCY6F8oBwYbwDvtEGNAEa2uO5RJ3OCStnDy7MNvHF89BtjdJME%2FfDtReuS4L7%2B%2B2X4JbRMbV7uF7AKnWJg4h3rW7R5QB42iCDqV%2Bw%2Fs5FaV77D05X1vXSisDcbILeqlJxnNHeGtocbxEA%2B0w71peJ0%2Bp3qGEn9NotNLhgHv6ieI3kTsR5POSnpCYmwptf%2BoUEfs%2FsN5kT7IkN%2BuDj6f29JBA4qE3Q%2F9TtDY5w%3D%3D&response-content-disposition=attachment%3B+filename%3Ddogs-vs-cats.zip'  --no-check-certificate -O cat_dog.zip
unzip cat_dog.zip
unzip train.zip
rm cat_dog.zip
rm train.zip
rm test1.zip

EBS 0.1 * 200 * 43200  /(86400 * 30) =