# Animal running trajectories analysis iterface
This app is destined for analysis of dogs running trajectories using Python's Pandas and [StatTools](https://gitlab.com/digiratory/StatTools) libs. Interface was created usinig *PyQt5*
Input - bunch of .csv files with trajectories data of some drugs, every drug must have several groups of dosage and control group
There are some examples of input data in Dogs_interface\trajectories
 Output - categorical plots of some movement parameters such as *Average speed*, *Total distance* ...  


## How to use
- Clone this repo and cd into dir
    ```
    git clone git@github.com:EcuaDora/Dogs_interface.git
    cd Dogs_interface
    ```
- Install dependencies
   ```
   pip install -r requirements.txt
   ```
- Run main.py file 
   ```
   python main.py
   ```
   or 
   ```
   python3 main.py
   ```
   If you are using some if Linux distributions, make sure that *Wayland* visual protocol is enabled, sometimes it is disabled by uncommenting line
   ```
   #WaylandEnable=false
   ```
   in  */etc/gdm3/custom.conf*
   If its so, comment this line back and reboot
   
   
## Project goals
Project is created in educational goals, by students of Sanit-Petersburg Electrotechnical University

   
![](https://media.tenor.com/xiII1Xqa0JAAAAAi/cachorro.gif)
