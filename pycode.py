def program(year, manufacturer, condition, fuel, odometer):
  # import libraries
  import pandas as pd
  import numpy as np
  # load dataframe
  df = pd.read_csv('./vehicles_michigan.csv')

  predictor1 = np.array(pd.to_numeric(df['year']), ndmin=2).T # year column into array
  predictor2 = pd.get_dummies(df['manufacturer'], drop_first=True) #manufacturer column to dummies values
  predictor3 = pd.get_dummies(df['condition'], drop_first=True) 
  predictor4 = pd.get_dummies(df['fuel'], drop_first=True) 
  predictor5 = np.array(pd.to_numeric(df['odometer']), ndmin=2).T 
  y = np.array(df['price'], ndmin=2).T 

  X = np.column_stack([np.ones(predictor1.shape), predictor1, predictor2, predictor3, predictor4, predictor5]) # predictors into one matrix
  XTX = np.dot(X.T, X) # Step 1
  XTX_inv = np.linalg.inv(XTX) # Step 2
  XTX_invXT = np.dot(XTX_inv, X.T) # Step 3
  w = np.dot(XTX_invXT, y) # least squares parameters

  x1 = np.array([year]) # input year 
  manufacturers = np.sort(df['manufacturer'].unique()) # get manufacturers list
  x2_arr = np.zeros(len(manufacturers)) # populate dummie values
  x2_arr[np.where(manufacturers == manufacturer)] = 1 # add 1 to the input manufacturer
  x2 = x2_arr[1:] 
  conditions = np.sort(df['condition'].unique())
  x3_arr = np.zeros(len(conditions))
  x3_arr[np.where(conditions == condition)] = 1
  x3 = x3_arr[1:] 
  fuels = np.sort(df['fuel'].unique())
  x4_arr = np.zeros(len(fuels))
  x4_arr[np.where(fuels == fuel)] = 1
  x4 = x4_arr[1:] 
  x5 = np.array([odometer])

  x = np.concatenate(([1], x1, x2, x3, x4, x5)) # get input values into one matrix
  price = np.dot(x, w) # calculate regression
  print("The car price is: $", end = '')
  print(f"{price[0]:,.2f}") # Should get over ~$40,000

def main():
  year = int(input("Enter the year of the car: ")) # Try '2018'
  manufacturer = input("Enter the manufacturer of the car: ") # Try 'tesla'
  condition = input("Enter the condition of the car: ") # Try 'good'
  fuel = input("Enter the fuel of the car: ") # Try 'electric'
  odometer = int(input("Enter the odometer of the car: ")) # Try '22000'

  program(year, manufacturer, condition, fuel, odometer)

main()
