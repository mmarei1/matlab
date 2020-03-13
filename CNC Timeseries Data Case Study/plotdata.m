% Plot figures
figure(1); clf reset;
plot(ts_interval,X_pre, 'r');
hold on;
plot(ts_interval,Y_pre, 'g');
plot(ts_interval,Z_pre, 'b');
ylabel('Displacements in x-y-z- axes (mm)');
xlabel('Duration (hours)');
legend('X displacement','Y displacement','Z displacement');
%% trace the cnc machine path in a 3d-plot
figure(2); clf reset;

plot3(X_pre,Y_pre,Z_pre);
hold on
headpos = plot3(X_pre(1),Y_pre(1), Z_pre(1),'o','MarkerFaceColor','r');

hold off;
title('CNC Machining Profile Motion')
xlabel('x (mm)');
ylabel('y (mm)');
zlabel('z (mm)');
dim = [0.6 0.3 0.2 0.1];
timestamp = ["time elapsed: ",string(t(1))];
a = annotation("textbox",dim,"str",timestamp);
legend('Line','CNC head position','Location','best');
for i = 1:size(X_pre,1)
    timestamp = ["time elapsed: ",string(t(i))];
    a.String(2) = {string(t(i))};
    headpos.XData = X_pre(i);
    headpos.YData = Y_pre(i);
    headpos.ZData = Z_pre(i);
    drawnow;
    m(i) = getframe;
end

% create a movie from the generated animation and play it once
save("machiningProfile.mat","m");


%% Pre-process data using several feature selection/extraction techniques
