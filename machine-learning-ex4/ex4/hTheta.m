% 此处为了便于自己理解，将课程中的符号进行了修改，对应如下
% input -> a1
% hiddens -> z2
% hiddens_sigmoid -> a2
% output -> z3
% outpu_sigmoid -> a3 即最终输出

function [hiddens,hiddens_sigmoid,h_theta] = hTheta(X,Theta1,Theta2)
  m = size(X,1);  
  
  % 把每一组数据转置成纵向，对应神经网络的图像，便于思考
  input = [ones(m,1) X];
  hiddens = Theta1 * input';
  hiddens_sigmoid = sigmoid(hiddens);
  
  output = Theta2 * [ones(1,m);hiddens_sigmoid];
  output_sigmoid = sigmoid(output);
  h_theta = output_sigmoid;
end