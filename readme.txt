
Please run the following steps in the Matlab command window:

1. a) Run Matlab in the directory MSCRA

   b) Run "Installmex.m" in the Matlab command window to generate some mex-files. 
      This step needs only to be performed once. 
   
   c) Run "Startup.m" in the Matlab command window.


2. a) To generate Figure 1 in the paper, perform:

      >> run_figure1;

   b) To generate Figure 2 in the paper, perform:

      >> run_figure2;

  c) To generate Figure 2 in the paper, perform:

      >> run_realdata;

   d) To generate the data in Table 2 in the paper, perform: 

      >> run_table2;

   e) To generate the data in Table 3 in the paper, perform: 

      >> run_table3;
  

Note: The numerical results obtained on different computers may be slightly different. 

Part of the subroutines and mexfiles in ``MSCRA'' are fetched from the Matlab package ``NNLS'' (available at http://www.math.nus.edu.sg/~mattohkc/NNLS.html)
written by Kim-Chuan Toh and Sangwoon Yun for their paper 
``An accelerated proximal gradient algorithm for nuclear norm regularized least squares problems, Pacific J. Optimization, 6 (2010), pp. 615--640''.


