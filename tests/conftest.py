import numpy as np
import pytest

@pytest.fixture
def toy():
    # One relation supports directed two-hop witnesses and explicit reciprocal facts.
    train=np.array([
       [0,0,1],[1,0,2],[0,0,3],[3,0,2],[4,0,5],[5,0,6],
       [2,0,0],[7,0,8],[8,0,9],[9,0,7],[1,1,4],[4,1,1],
       [1,1,6],[6,1,1],[5,1,2],[2,1,5],[7,1,0],[8,1,2],
       [2,2,3],[3,2,0],[6,2,7],[0,2,6],[2,2,5],[9,2,1],
    ],dtype=np.int64)
    val=np.array([[2,1,7],[7,2,1]],dtype=np.int64)
    test=np.array([[0,1,2],[1,2,8]],dtype=np.int64)
    return train,val,test,10,3
