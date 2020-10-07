
The basic pytorch architecture was based on https://www.youtube.com/watch?v=9j-_dOze4IM by (https://github.com/Sentdex), but subsequent modification from classification network to an auto-encoder performed by John Balis. 

## usage 


comparison between original and encoded -> decoded samples

`python3 ./auto_encoder.py torch comparison`

latent space traversal between points in encoded space

`python3 ./auto_encoder torch traversal` 

The helper bash script is for realtime traversal, it's a hack to close the pyplot window over and over again. You need wmctrl installed for it to work.
