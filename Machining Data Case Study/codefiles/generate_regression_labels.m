% Calculate the Coefficient C for each experimental run

%clear all; close all; clc;
set(0,'DefaultFigureWindowStyle','docked')
% calculate the max time achieved in each experimental run
load cuttingParametersTable.mat

vb = 0.4; % flank wear threshold

for i = 1:5
    cs(i,:) = cuttingSpeeds(i)*ones(1,5);
end

% reshape the data into a column vector
vc = reshape(cs',[25,1]);

for i=1:25
    % find the max cutting length
    [cl_val,cl_i] = max(table2array(machiningDataTable(i,2:end)));
    cuttingLength = lengthIntervals(cl_i);
    t_max = cuttingLength/vc(i);
    cuttingTimes(i) = t_max;
end

% the max cutting length occurs at a point that exceeds 
% the true maximum cutting length attainable from the tool.
% To estimate the true cutting value, we need to interpolate between the
% max value and the max-1 value

% to analyse experiment 14 (with the longest cutting tool life record
plots = false;
for i = 1:25
    yvalues = table2array(machiningDataTable(i,2:end));
    [val,idx] = max(yvalues);
    ydata = [0 yvalues(1:idx)];
    maxIndex = size(ydata,2);
    xdata = [0 lengthIntervals(1:maxIndex-1)];
    xmax = lengthIntervals(maxIndex-1);
    % plot(cuttingTimes,ydata,'bo','LineWidth',2,'MarkerSize',10)
    %plot([xdata max(xdata)+10], vb*ones(1,maxIndex+1),'r-.');
    xq = interp1(ydata,xdata,0.4);
    % if xq is NaN, we extrapolate using the next method instead
    if isnan(xq)
      xq = interp1(ydata,xdata,0.5,'nearest','extrap');
      % calculate additional offset from extrapolating the gradient
      % "dirty" gradient because it assumes yvalue variation is constant
      dy = (0.5 - val)/(xq - lengthIntervals(idx-1));
      xd = dy*xq;
      xq = xq + xd;
      val = 0.4;
      xmax = xq;
    end
    l_vb(i) = xq;
    
    max_wearValue(i) = val;
    distances{i} = sort([xdata, xq]);
    wear_values{i} = sort([ydata, 0.4]);
    cuttingTimeValues{i} = calculateTime(distances{i},vc(i));
    
    if plots==true
    % Plot the results of the interpolation along with the wear profile
        figure(i); clf reset
        plot(xdata,ydata,'bo','LineWidth',2,'MarkerSize',10)
        hold on;
        plot([xdata max(xdata)+30], vb*ones(1,maxIndex+1),'r-.','LineWidth',2)
        hold on
        plot(xq,0.4,'kx','LineWidth',2,'MarkerSize',10)
        title(sprintf("Exp %2g: l_V_B =%4g m",i,xq),'FontSize',18)
        axis([0 xq+10 0 val])
        xlabel('Cutting length (m)','FontSize',18);
        ylabel('V_B (mm)','FontSize',18);
        legend('Measured wear values','V_B','failure at V=V_B','FontSize',16,'Location','NorthEastOutside')
        % Save the figures and the estimated max cutting length
        expName = ['exp',sprintf('%02d',i),'profile.png'];
        saveas(gcf,expName);
        % expNamefig = ['exp',sprintf('%02d',i),'profile.fig'];
        % saveas(gcf,expNamefig);    
    end

end

function t = calculateTime(distance,speed)
   % helper function to calculate time of cutting from speed and cutting distance  
   %
   % convert speed from mm/min to m/s
   speed = speed/(60*10-3);
   t = (1/speed).*distance;
end
