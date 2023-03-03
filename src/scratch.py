#%%
import gurobipy as gp
import numpy as np
from gurobipy import GRB

#model 2d gradient chart with x values.
import matplotlib.pyplot as plt
import seaborn as sns

# Create a Model instance
m = gp.Model()

# Add variables
E = np.loadtxt("exps.txt", dtype=int )

X = m.addMVar( (5, 7), vtype=GRB.INTEGER, name="X" )
W = m.addMVar( (5, 7), vtype=GRB.BINARY, name="W" )



# Providing the coefficients and the sense of the objective function
m.setObjective( sum( X[i][j] for i in range( 5 ) for j in range( 7 ) ), GRB.MINIMIZE)

# Adding the 6 constraints
c1 = m.addConstrs( 60*W[i][j] <= X[i-1][j] + X[i+1][j] + X[i][j-1] + X[i][j+1] for i in range( 1, 4 ) for j in range( 1, 6 ) )  # center
c2 = m.addConstrs( 45*W[0][j] <= X[1][j] + X[0][j-1] + X[0][j+1] for j in range(1, 6) )  # up
c3 = m.addConstrs( 45*W[4][j] <= X[3][j] + X[4][j-1] + X[4][j+1] for j in range(1, 6) )  # down
c4 = m.addConstrs( 45*W[i][0] <= X[i][1] + X[i-1][0] + X[i+1][0] for i in range(1, 4) )  # left
c5 = m.addConstrs( 45*W[i][6] <= X[i][5] + X[i-1][6] + X[i+1][6] for i in range(1, 4) )  # right
c6 = m.addConstr( 30*W[0][0] <= X[1][0] + X[0][1] ) # upper left corner
c7 = m.addConstr( 30*W[4][0] <= X[3][0] + X[4][1] ) # bottom left corner
c8 = m.addConstr( 30*W[4][6] <= X[3][6] + X[4][5] ) # bottom right corner
c9 = m.addConstr( 30*W[0][6] <= X[1][6] + X[0][5] ) # upper right corner

c10 = m.addConstrs( W[i][j] <= W[i-1][j] + W[i+1][j] + W[i][j-1] + W[i][j+1] for i in range( 1, 4 ) for j in range( 1, 6 ) )    # center
c11 = m.addConstrs( W[0][j] <= W[1][j] + W[0][j-1] + W[0][j+1] for j in range(1, 6) )  # up
c12 = m.addConstrs( W[4][j] <= W[3][j] + W[4][j-1] + W[4][j+1] for j in range(1, 6) )  # down
c13 = m.addConstrs( W[i][0] <= W[i][1] + W[i-1][0] + W[i+1][0] for i in range(1, 4) )  # left
c14 = m.addConstrs( W[i][6] <= W[i][5] + W[i-1][6] + W[i+1][6] for i in range(1, 4) )  # right
c15 = m.addConstr( W[0][0] <= W[1][0] + W[0][1] ) # upper left corner
c16 = m.addConstr( W[4][0] <= W[3][0] + W[4][1] ) # bottom left corner
c17 = m.addConstr( W[4][6] <= W[3][6] + W[4][5] ) # bottom right corner
c18 = m.addConstr( W[0][6] <= W[1][6] + W[0][5] ) # upper right corner

c19 = m.addConstrs( E[i][j] - X[i][j] <= 30 * (1-W[i][j]) for i in range ( 5 ) for j in range( 7 ) )

c20 = m.addConstr( sum( W[i][j] for i in range( 5 ) for j in range( 7 ) ) >= 18 )

c21 = m.addConstrs( X[i][j] >= 0 for i in range( 5 ) for j in range( 7 ) )

# Solving the model
m.optimize()

m.printAttr('X') #This prints the non-zero solutions found


x_arr = m.getAttr('X')
two_D_X_chart = np.arange(35).reshape(5, 7)

for i in range (5):
    for j in range (7):
        two_D_X_chart[i][j] = x_arr[(7 * i) + j]


print(two_D_X_chart)

sns.heatmap(two_D_X_chart, annot=True, annot_kws={"size": 7})



#%%
import gurobipy as gp
import numpy as np
from gurobipy import GRB


#############################################################
#                       PART 2                              #
#############################################################

m2 = gp.Model()

S = m2.addMVar( (7, 5, 7), vtype=GRB.BINARY, name="S")
#R = m2.addMVar( 7, vtype=GRB.BINARY ,name="R")
R = m2.addVars(7, vtype=GRB.BINARY, name="R")
Z = m2.addMVar( (1, 7), vtype=GRB.BINARY, name="Z" )
W = np.loadtxt( "w_values.txt", dtype=int )

# Objective

m2.setObjective( sum( R[i] for i in range (7) ), GRB.MAXIMIZE )

# Constraints
c2_1 = m2.addConstrs( ( 3 - ( sum(S[k][i][j] * W[i][j] for i in range(5) for j in range(7) ) ) <= 3*Z[0][k] ) for k in range( 7 ) )
c2_2 = m2.addConstrs( ( R[k] <= 3*(1-Z[0][k]) ) for k in range( 7 ) )


c2_3 = m2.addConstrs( ( sum( S[k][i][j] for k in range( 7 ) ) == 1 ) for i in range(5) for j in range(7) )

c2_4 = m2.addConstrs( ( S[4][i][0] == 1 ) for i in range(5) )        # west border region
c2_5 = m2.addConstrs( ( S[5][0][j] == 1 ) for j in range(1, 6) )     # north border region
c2_6 = m2.addConstrs( ( S[6][i][6] == 1 ) for i in range(5) )        # east border region

c2_7 = m2.addConstrs( ( S[k][i][j] <= S[k][i-1][j] + S[k][i+1][j] + S[k][i][j-1] + S[k][i][j+1] ) for i in range( 1, 4 ) for j in range( 1, 6 ) for k in range(7) )  # center
c2_8 = m2.addConstrs( S[k][0][j] <= S[k][1][j] + S[k][0][j-1] + S[k][0][j+1] for j in range(1, 6) for k in range(7) )  # up
c2_9 = m2.addConstrs( S[k][4][j] <= S[k][3][j] + S[k][4][j-1] + S[k][4][j+1] for j in range(1, 6) for k in range(7) ) # down
c2_10 = m2.addConstrs( S[k][i][0] <= S[k][i][1] + S[k][i-1][0] + S[k][i+1][0] for i in range(1, 4) for k in range(7) ) # left
c2_11 = m2.addConstrs( S[k][i][6] <= S[k][i][5] + S[k][i-1][6] + S[k][i+1][6] for i in range(1, 4) for k in range(7) ) # right
c2_12 = m2.addConstrs( ( S[k][0][0] <= S[k][1][0] + S[k][0][1] ) for k in range(7) )    # upper left corner
c2_13 = m2.addConstrs( ( S[k][4][0] <= S[k][3][0] + S[k][4][1] ) for k in range(7) )    # bottom left corner
c2_14 = m2.addConstrs( ( S[k][4][6] <= S[k][3][6] + S[k][4][5] ) for k in range(7) )    # bottom right corner
c2_15 = m2.addConstrs( ( S[k][0][6] <= S[k][1][6] + S[k][0][5] ) for k in range(7) )    # upper right corner

# Region A
c2_16 = m2.addConstr( S[0][3][1] == 1 )
c2_17 = m2.addConstr( S[0][4][1] == 1 )
c2_18 = m2.addConstr( S[0][3][2] == 1 )
c2_19 = m2.addConstr( S[0][4][2] == 1 )


# Region B
c2_20 = m2.addConstr( S[1][1][1] == 1 )
c2_21 = m2.addConstr( S[1][1][2] == 1 )
c2_22 = m2.addConstr( S[1][1][3] == 1 )
c2_23 = m2.addConstr( S[1][2][2] == 1 )

# Region C
c2_24 = m2.addConstr( S[2][1][4] == 1 )
c2_25 = m2.addConstr( S[2][1][5] == 1 )

# Region D
c2_26 = m2.addConstr( S[3][3][4] == 1 )
c2_27 = m2.addConstr( S[3][4][4] == 1 )
c2_28 = m2.addConstr( S[3][4][5] == 1 )

c2_29 = m2.addConstrs( S[1][4][i] == 0 for i in range (1,6))
c2_30 = m2.addConstrs( S[2][4][i] == 0 for i in range (1,6))
#c2_31 = m2.addConstrs( sum(S[0][4][i])  >=2 for i in range (1,6))

m2.optimize()



#%%

import gurobipy as gp
import numpy as np
from gurobipy import GRB


# Create a Model instance
m3 = gp.Model()

# Add variables
S = m3.addMVar( (7, 5, 7), vtype=GRB.BINARY, name="S" )
R = m3.addVars( 7, vtype=GRB.BINARY, name="R" )
Z = m3.addMVar( (1, 7), vtype=GRB.BINARY, name="Z" )
W = np.loadtxt( "w_values.txt", dtype=int )

# Objective
m3.setObjective( sum( R[i] for i in range (7) ), GRB.MAXIMIZE )

# Constraints
c3_1 = m3.addConstrs( ( 3 - ( sum(S[k][i][j] * W[i][j] for i in range(5) for j in range(7) ) ) <= 3*Z[0][k] ) for k in range( 7 ) )
c3_2 = m3.addConstrs( ( R[k] <= 3*(1-Z[0][k]) ) for k in range( 7 ) )

c3_3 = m3.addConstrs( ( sum( S[k][i][j] for k in range( 7 ) ) == 1 ) for i in range(5) for j in range(7) )

c3_4 = m3.addConstrs( ( S[4][i][0] == 1 ) for i in range(5) )        # west border region
c3_5 = m3.addConstrs( ( S[6][i][6] == 1 ) for i in range(5) )        # east border region

c3_6 = m3.addConstrs( ( S[k][i][j] <= S[k][i-1][j] + S[k][i+1][j] + S[k][i][j-1] + S[k][i][j+1] ) for i in range( 1, 4 ) for j in range( 1, 6 ) for k in range(7) )  # center
c3_7 = m3.addConstrs( S[k][0][j] <= S[k][1][j] + S[k][0][j-1] + S[k][0][j+1] for j in range(1, 6) for k in range(7) )  # up
c3_8 = m3.addConstrs( S[k][4][j] <= S[k][3][j] + S[k][4][j-1] + S[k][4][j+1] for j in range(1, 6) for k in range(7) ) # down
c3_9 = m3.addConstrs( S[k][i][0] <= S[k][i][1] + S[k][i-1][0] + S[k][i+1][0] for i in range(1, 4) for k in range(7) ) # left
c3_10 = m3.addConstrs( S[k][i][6] <= S[k][i][5] + S[k][i-1][6] + S[k][i+1][6] for i in range(1, 4) for k in range(7) ) # right
c3_11 = m3.addConstrs( ( S[k][0][0] <= S[k][1][0] + S[k][0][1] ) for k in range(7) )    # upper left corner
c3_12 = m3.addConstrs( ( S[k][4][0] <= S[k][3][0] + S[k][4][1] ) for k in range(7) )    # bottom left corner
c3_13 = m3.addConstrs( ( S[k][4][6] <= S[k][3][6] + S[k][4][5] ) for k in range(7) )    # bottom right corner
c3_14 = m3.addConstrs( ( S[k][0][6] <= S[k][1][6] + S[k][0][5] ) for k in range(7) )    # upper right corner

m3.optimize()


#%%
