# An Integer-Only Resource-Minimized RNN on FPGA for Low-Frequency Sensors in Edge-AI

**This Repository is a description and tutorial on how adapt and use an Integer-Only Resource-Minimized Recurrent Neural Network Implementation on FPGA. It describes the Tensorflow to Integer-only model conversion and how to use the full HDL implementation. The source files can be found on Zenodo: https://zenodo.org/record/7800728. We also recommend you to read the recently published journal article on this work: https://ieeexplore.ieee.org/iel7/7361/4427201/10161725.pdf**

This repository offers the code for a Recurrent Neural Network Implementation on FPGA, referred to as Integer-Only Resource-Minimized Recurrent Neural Network (RNN), along with a comprehensive guide on its usage in a few easy steps, making it easy to use in sensor applications. Currently, a scientific work disclosing the full details of this RNN is under review, and will be added as a reference in a future date. The RNN is built using one or multiple simple RNN layers followed by a linear layer, utilizing Tensorflow 2.0 (Keras Layers). The guide consists of two parts:

1.  Python: Tensorflow 2.0 model to integer-only shared-scale RNN conversion and memory extraction.
    
2.  HDL: Synthesis in Lattice Radiant and FPGA implementation.
    

! -- Caution -- !

All RNN layers have to be Keras-based "simple RNN" with an equal width, i.e., the weight dimensions have to be equivalent for each layer. Only a single linear layer that follows the simple RNN layers is supported. The following is an example model description from Tensorflow 2.0 (cow behavior estimation, see model description in Ref. \[1\], dataset openly available at Ref. \[2\]), here the parameters are {_N, L, I, O, q, Ts, fs_}: {13, 4, 3, 4, 8, 35, 25}:

     Layer (type)                Output Shape              Param #   
    =================================================================
     simple_rnn_8 (SimpleRNN)    (None, 35, 13)            221       
                                                                     
     simple_rnn_9 (SimpleRNN)    (None, 35, 13)            351       
                                                                     
     simple_rnn_10 (SimpleRNN)   (None, 35, 13)            351       
                                                                     
     simple_rnn_11 (SimpleRNN)   (None, 13)                351       
                                                                     
     dense_2 (Dense)             (None, 4)                 56        
                                                                     
    =================================================================
    Total params: 1,330
    Trainable params: 1,330
    Non-trainable params: 0
    _______________________

# 1. Python 
**Install pip requirements**

Before you use any of the Python code, make sure you install the requirements.txt in the Python project folder:

    $ cd ./python
    $ pip install -r requirements.txt

**Tensorflow 2.0 SimpleRNN --> Integer-Only Shared-Scale RNN and ".mem" files for FPGA impl.:**

First you have to set the parameters of your simpleRNN in the "parameters.csv" file, located in the top "python" folder. Please check the section about parameters what parameters you have to set. After setting parameters, you have to make sure that the x dataset and y dataset (input data and labels) are saved as .npy files (x\_test.npy, y\_test.npy) files. Then, you can run the code from the "python" directory:

    $ python3 ./src/convert_and_extract.py --parameter_file parameters.csv --model_directory TF_model_directory --x_data directory_to_x_test.npy --y_data directory_to_y_test.npy

Example of cow behavior estimation (default setting):

Change directory to "python" and run:

    $ python3 ./src/convert_and_extract.py --parameter_file ./cow/parameters.csv --model_directory ./cow/model --x_data ./cow/x_test.npy --y_data ./cow/y_test.npy
    

output

    50/50 [==============================] - 1s 6ms/step
    100%|██████████████████████████████████████| 1594/1594 [00:09<00:00, 167.04it/s]
    pre-quantized model (TF2.0) ---
    Misclassified labels :77
    top-1 accuracy: 95.17
    
    post-quantized model (shared-scale RNN) ---
    Misclassified labels :78
    top-1 accuracy: 95.11 %
    

If you do not have the test data at hand, you can leave out --x\_data and --y\_data and the Python code will use random data to convert the model. This will not change anything however you will not be able to see how much the accuracy drops when converting to 8-bit integer on the shared-scale RNN. By running the above code, the to 8-bit integer converted model weights are automatically copied to the "HDL" folder as "RNN\_8b.mem.

**Parameter setting**

The following list of parameters has to be defined by you, based on your own pre-trained model:

*   _N_: Layer width
*   _L_: number of RNN layers
*   _I_: number of inputs
*   _O_: number of classes
*   _Ts_: number of timesteps
*   _fs_: sensor frequency --> necessary for setting minimum clock frequency of FPGA

At the moment only 8-bit bitwidth is supported!

Adjust these parameters according to your own model in "parameters.csv" before running the convert\_and\_extract.py file in the above order (see "parameters.csv" in the cow folder for an example). For the HDL, you need to adjust the "RNNCommon.sv" file (only 6 lines at the top):

localparam LAYERS (_L_), LAYER\_WIDTH (_N_), INPUT\_WIDTH (_I_), TIMESTEPS (_Ts_), CLASS\_COUNT (_O_), INPUT\_FREQ (_fs_)

# 2\. HDL and implementation on Lattice FPGA

**Module Tree**

    ├── top - top.sv
    │   ├── IOController - IOController.sv --> SPI Controller (Slave)
    │   │   └── SPI_Slave - SPI_Slave.sv --> Please see SPI_Slave.v section
    │   ├── RNN - RNN.sv --> top module of RNN
    │   │   ├── VecRAM - VecRAM.sv --> RAM wrapper for storing recurrent vectors (h_t)
    │   │   │   └── pmi_ram_dq - pmi_ram_dq.v --> Lattice RAM IP
    │   │   ├── TimestepController - TimestepController.sv --> FSM of the RNN (central module)
    │   │   ├── Tanh - Tanh.sv --> 1-to-1 Hyperbolic tangent mapping
    │   │   │   └── pmi_rom - pmi_rom.v --> Lattice ROM IP
    │   │   ├── RNNParam - RNNParam.sv --> contains RNN model weights and parameters
    │   │   │   └── ParamROM - pmi_rom.v   --> model weight wrapper
    │   │   │       └──pmi_rom - pmi_rom.v --> Lattice ROM IP
    │   │   ├── ALU - ALU.sv --> performs 3 operations relevant to RNN
    │   │   │   ├── 2x pmi_sub - pmi_sub.v   --> Lattice substracter IP
    │   │   │   ├── 2x pmi_mult - pmi_mult.v --> Lattice multipier IP
    │   │   │   └── 3x pmi_add - pmi_add.v   --> Lattice adder IP

Information about the IPs and how to use them can be found here (hyperlink might not work, please copy into browser):

Arithmethic modules: [http://www.latticesemi.com/view\_document?document\_id=52684](http://www.latticesemi.com/view_document?document_id=52684)

Memory modules: [http://www.latticesemi.com/view\_document?document\_id=52685](http://www.latticesemi.com/view_document?document_id=52684)

**Synthesis**

After setting the parameters according to your own model in "RNNCommon.sv" the following two steps have to be performed for FPGA implementation:

*   Open project with "RNN\_f.rdf" on Lattice Radiant, set the Lattice FPGA that you need (default is set to Lattice ICE40UP5K) or check "**Adaptation for other FPGAs**" section below. Used Lattice Radiant Ver. 3.2.0.18.0.
*   Adjust the constraint files, currently the following signals are set to these pins on Lattice ICE40UP5K, these pins are located on bank 2 of the ICE40UP5K-B-EVN breakout board using SGI48, adjust according to own settings:
    *   clk\_ext (external clock) --> 35 (12 MHz clock of breakout board)
    *   sclk --> 44 (Bank2, SPI)
    *   mosi --> 47 (Bank2, SPI)
    *   ss --> 46 (Bank2, SPI)
    *   miso --> 45 (Bank2, SPI)
    *   rst\_ext --> 23 (button on the breakout board)

**Running the model on FPGA with a data stream**

After implementing on FPGA, first reset the FPGA with rst\_ext. Hereafter, the RNN automatically starts when you send _I_ single byte SPI messages where the byte contains one data point of the input data. Make sure you repeat transmission of _I_ bytes with SPI for the set amount of timesteps, _Ts,_ and make sure to leave enough time in between (equivalent to that of the actual sensor sampling frequency). When you complete transmission for the set amount of timesteps, send _O_ individual single byte dummy messages, the slave FPGA will then encode the classification results unto these messages.

# 3\. Adaptation for other FPGAs
The supported FPGAs are only those by Lattice (Lattice Semiconductor Inc., Hillsboro OG) because IP modules from this vendor are used. However, if you change the following IPs, Xilinx, Intel FPGAs etc. are compatible:  
\- pmi\_rom --> "RNNParam.sv", "Tanh.sv"  
\- pmi\_ram\_dq --> "VecRAM.sv"  
\- pmi\_add, pmi\_sub, pmi\_mult --> "ALU.sv"  
We cannot guarantee that the implementation will work for other vendors. However, please contact us if you have any idea for collaboration or for any questions!

**SPI\_Slave.v (MIT License, https://github.com/nandland/spi-slave)**

A softcore SPI\_Slave module has been utilized to process transmission of messages from the Master. This module, written in verilog, was retrieved from https://github.com/nandland/spi-slave. This module is under a separate license, the MIT License. The License and copyright details have been added as a header of SPI\_Slave.v as necessary and as a seperate file ("LICENSE") under the HDL/source/SPI\_Slave directory. Please note that the other source files of this project are copyrighted and owned (Excluding the Lattice IPs) by the authors of this repository, licensed under the GNU GENERAL PUBLIC LICENSE, see "LICENSE" file in the root directory for more details.

**References**

\[1\]: [J. Bartels, K. K. Tokgoz, S. A, M. Fukawa, S. Otsubo, C. Li, I. Rachi, K.-I. Takeda, L. Minati, and H. Ito, "Tinycownet: Memory- and powerminimized rnns implementable on tiny edge devices for lifelong cow behavior distribution estimation," IEEE Access, vol. 10, pp. 32 706– 32 727, 2022.](https://ieeexplore.ieee.org/abstract/document/9726221)  
\[2\]:[H. Ito, K. Takeda, K. Tokgoz, L. Minati, M. Fukawa, L. Chao, J. Bartels, I. Rachi, and A. Sihan, “‘japanese black beef cow behavior classification dataset,” 2022.](https://zenodo.org/record/5849025#.ZCFKJ9JBxR7)
