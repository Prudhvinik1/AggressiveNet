# AggressiveNet

### Libraries perquisites:
- python 3.5
- numpy 1.14.0
- keras 2.2.0
- tensorflow 1.9.0
- Pillow 3.1.2
- opencv-python 3.4.1.15 (Try pip install opencv-python or conda install -c menpo opencv3 if it doesn't work)

### DataSets

* [UCF CRIMES](https://webpages.uncc.edu/cchen62/dataset.html)
* [Hockey](http://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89)
* [Movies](http://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)
* [UniCrimes](http://didt.inictel-uni.edu.pe/dataset/UNI-Crime_Dataset.rar)
* [ViolentFlows](https://www.openu.ac.il/home/hassner/data/violentflows/)


### How to run

#### ConvLSTM model
- To run ConvLSTM model on the datasets, run the file run.py in ConvLSTM folder. You can select and change your desired architectures and training parameters in run.py file.
#### LSTM and Bidirectional LSTM model
- To run LSTM model on the datasets, run the file crime_vgg16_lstm_hi_dim_train.py in LSTM/demo folder. This will run VGG+VGGLSTM model on ucfcrimes dataset. 
- To run BidirectionalLSTM model on the datasets, run the file crime_vgg16_bilstm_hi_dim_train.py in LSTM/demo folder. This will run VGG+VGGBiLSTM model on ucfcrimes dataset. 
- To run LSTM model on the datasets with resnet, run the file crime_resnet50_lstm_hi_dim_train in LSTM/demo folder. This will run Resnet50+LSTM model on ucfcrimes dataset. 
#### Object detection
- To run handgun object_detection ipynb file, follow the cell-by-cell instructions and run the ipynb file after setting up data as mentioned in the notebook. It is preferable to run it on Google Colab.

