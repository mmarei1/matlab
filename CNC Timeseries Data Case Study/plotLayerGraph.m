function plotLayerGraph(lgraph,ldetails)
    % get the layers and connections of the lgraph
    layers = lgraph.Layers;
    connections = lgraph.Connections;
    xscale = 0.95;
    spacing = 0.04;
    n=numel(layers);
    % shift all x-coordinates by the width of the 
    %xShift = xscale*(((0.07+2*spacing*n)-(0.11+2*spacing*1)))/2
    % calculate horizontal shift from the number of 2*spacingss -1
    xShift = (1-xscale*((spacing*(2*n+2))))/2;
    figure;
    plot(-0.5:0.05:3,'w');
    hold on;
    for l = 1:n
        % manipulate string and replace the underscores in it with spaces
        lyr = layers(l);
        s = layers(l).Name;
        s = replace(s,"_"," ");
        r =1;
        if (isa(lyr,'nnet.cnn.layer.SequenceInputLayer'))
            s = sprintf("Sequence Input Layer\n\nInput Size:\n%d",layers(l).InputSize);
            r = 1;
            vOff = 0;
            fcolor = [0 0.4470 0.7410];
        elseif(isa(lyr,'nnet.cnn.layer.LSTMLayer'))
            s = sprintf("%s\n\n# Hidden Units: \n%d",s,layers(l).NumHiddenUnits);
            lsz = layers(l).NumHiddenUnits;
            plsz = layers(2).NumHiddenUnits;
            %fprintf("Layer %d size = %d \n",l,lsz)
            %fprintf("Previous FC Layer %d size = %d \n",l,plsz)
            r = lsz/plsz;
            vOff = 0.1*(1-r)/2^r;
            fcolor = [0.4940 0.1840 0.5560]	;
        elseif(isa(lyr,'nnet.cnn.layer.FullyConnectedLayer'))
            %s = {s,"Output Size: ",layers(l).OutputSize};
            s = sprintf("%s\n\n Output Size: \n%d",s,layers(l).OutputSize);
            lsz = layers(l).OutputSize;
            plsz = layers(2).OutputSize;
            r = lsz/plsz;
            vOff = 0.1*(1-r)/2^r;
            fcolor = [0.4660 0.6740 0.1880]	;
        elseif(isa(lyr,'nnet.cnn.layer.RegressionOutputLayer'))
            s = sprintf("Regression Output Layer\n\nLoss Function:\nMSE");
            r = 1;
            vOff = 0;
            fcolor = [0.8500 0.3250 0.0980]	;
        else
            s = {s,"Layer"};
            r = 1
            vOff = 0;
        end
        
        relHeight = 0.25*r+0.4;
        pos = [xscale*(2*spacing*l)+(xShift), 0.2+vOff, xscale*spacing, relHeight];
        tbx = annotation('textbox');
        tbx.FontSize = 10;
        tbx.Position = pos;
        tbx.String = s;
        tbx.HorizontalAlignment = 'center';
        tbx.VerticalAlignment = "middle";
%         tbx.FaceColor = fcolor;
%         tbx.FaceAlpha = 0.2;
        rect = annotation("rectangle");
        rect.Position = pos;
        rect.FaceColor = fcolor;
        rect.FaceAlpha = 0.2;
        %drawnow
    end
    %drawnow
    for c = 1:0.5*numel(connections)
        % draw an arrow
        xpos = [xscale*(spacing+2*c*spacing)+(xShift),xscale*(2*spacing+2*c*spacing)+(xShift)];
        ypos = [0.5,0.5];
        annotation("textarrow",xpos,ypos,'LineWidth',1);     
    end
    drawnow
    hold off;
    title(sprintf(ldetails))
end