function layers = unLearnWeights(layers,factor)

for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'Weights$', 'once'))
            layers(ii).(propName) = factor.*layers(ii).(propName);
        end
        if ~isempty(regexp(propName, 'Bias$', 'once'))
            layers(ii).(propName) = factor.*layers(ii).(propName);
        end
    end
end

end