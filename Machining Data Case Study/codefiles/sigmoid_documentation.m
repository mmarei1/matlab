%% |sigmoid| documentation 
% This function evaluates a simple sigmoid function along _x_ such that 
% 
% <<sigmoid_documentation_eq.png>>
% 
%% Syntax 
% 
%  y = sigmoid(x) 
%  y = sigmoid(x,c)
%  y = sigmoid(x,c,a)
% 
%% Description
% 
% |y = sigmoid(x)| generates a sigmoid function along |x|. 
% 
% |y = sigmoid(x,c)| makes a sigmoid that scaled from zero to one, where
% c corresponds to the |x| value where y = 0.5. If |c| is not specified, a
% default value of |c = 0| is assumed. 
% 
% |y = sigmoid(x,c,a)| specifies |a|, the rate of change. If |a| is close to
% zero, the sigmoid function will be gradual. If |a| is large, the sigmoid
% function will have a steep or sharp transition. If |a| is negative, the 
% sigmoid will go from 1 to zero. A default value of |a=1| is assumed if 
% |a| is not declared. 
% 
%% Example 1 
% A simple sigmoid: 

x = -10:.01:10;
plot(x,sigmoid(x))

%% Example 2
% Make a sigmoid function along x = 1 to 100, such that y(x=60) = 0.5: 

x = 1:100; 
y = sigmoid(x,60);

figure
plot(x,y,'b','linewidth',2) 
box off

%%
% Now do the same thing as above, but make the transition more gradual: 

y2 = sigmoid(x,60,0.1); 
hold on
plot(x,y2,'r','linewidth',2) 
legend('default a = 1','a = 1/10','location','northwest') 
legend boxoff 

%% Author Info:
% This function was written by <http://www.chadagreene Chad A. Greene>
% of the University of Texas Institute for Geophysics (UTIG), May 28, 2015.  