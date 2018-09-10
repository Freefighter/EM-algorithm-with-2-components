# EM-algorithm-with-2-components

[This package](https://github.com/Freefighter/EM-algorithm-with-2-components) is a tool for estimating the parameters in a mixtrue model with 2 components. We extend the basic model so it can not only deal with GMM but also Poisson MM. Indeed, the users are allowed to customize their own distribution by passing the specific E-Step, M-Step, likelyhood function, and Infomation Matrix.

# Statistical Model

Y is affected by Z and hidden variable \Delta \in \lbrace 0, 1 \rbrace, so there are two "kinds" of Y.

And \Delta is affected by X.

# Structure of the package

©¦ demo.py
©¦
©À©¤EM_algo
©¦  ©¦  __init__.py
©¦  ©¦
©¦  ©À©¤EM_func
©¦  ©¦  ©¦  GaussianModel.py
©¦  ©¦  ©¦  PoissonModel.py
©¦  ©¦  ©¦  __init__.py
©¦  ©¦  ©¦
©¦  ©¦  ©¸©¤__pycache__
©¦  ©¦          GaussianModel.cpython-36.pyc
©¦  ©¦          PoissonModel.cpython-36.pyc
©¦  ©¦          __init__.cpython-36.pyc
©¦  ©¦
©¦  ©¸©¤__pycache__
©¦          __init__.cpython-36.pyc
©¦
©¸©¤__pycache__
        EM_algo.cpython-36.pyc
        EM_func.cpython-36.pyc
        
I put the basic procedure in EM_algo, and prepare the functions related to Gaussian and Poisson distribution in EM_algo.EM_func.


### Methods

pass

### How to Use

pass

# License

    Copyright 2018 Yifan Chen
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
