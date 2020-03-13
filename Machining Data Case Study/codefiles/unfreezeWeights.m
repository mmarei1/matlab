% layers = unfreezeWeights(layers, lrf) 
% returns the model layers designated which have a settable weight/bias learning rate property
% with a specified factor of the overall model learning rate.
% Adapted by Mohamed Marei (08/11/2019)
function layers = unfreezeWeights(layers,lrf)
    % increase the weight/bias learn rate factor 
for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = lrf;
        end
    end
end

end