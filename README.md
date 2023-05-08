This is an implementation of my Master dissertation [Hybrid Model of Ballistics](https://haolinwang2001.github.io/files/HybridModelOfBallistics.pdf).  
### **How To Use**  
Training data and testing data should be in the folder `data`.  
`python main.py --cfg file_for_training.csv --test file_for_testing.csv`  
  
For people don't like command lines, you could open `JustRun.py` in your favourate IDE(e.g. Spyder), click the Run button and follow the instructions.   
`JustRun.py` file also provides the function of filling the predict velocity to a csv file. Fill a template in folder `data/fillin` and store the csv file under this directory, this script will generate a csv file with predicted velocity of bullets for you. 

### **File Descriptions**  
- `cdg.py`: return drag coefficient number according to velocity.
- `deapcalc.py`: calculations that are used in DEAP training.  
- `prepdata.py`: prepare the data for training. Also include the function for testing.
- `velocity.py`: calculate scalar velocity at a certain distance given initial velocity and other factors.
