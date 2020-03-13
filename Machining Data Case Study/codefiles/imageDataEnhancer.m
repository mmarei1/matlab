classdef imageDataEnhancer < imageDataAugmenter
    % ImageDataEnhancer object for further enhancing data
    % 
    % imageDataEnhancer object implements further transformations,
    % including batch-cropping the image to custom-size blocks, mean
    % subtraction, smoothing, and subregion isolation of high-variance
    % feature regions.
    % 
    % Additional Properties:
    % 
    % Subcrops - number of sub-images to divide image into
    % Sigma    - Value dictating Gaussian Kernel smoothing coefficient
    % 
    % 
    % Copyright 2018 - Mohamed Marei
    %
    % Base Class implemented by The Mathworks, Inc .
    properties(SetAccess = 'private')
        Subcrops
        
    end
    
    properties (Hidden)
        numberOfSubcrops
    end
    
    methods
        % start of constructor/initializer
        function self = imageDataEnhancer(varargin)
            self.parseInputs(varargin{:});
            self.Stream = RandStream('mt19937ar','Seed',0);
            
            self.isRandScaleSpecified = false;
            if ~isequal(self.RandScale,[1 1])
                self.isRandScaleSpecified = true;
            end
            
            %Error if RandXScale or RandYScale specified along with
            %RandScale
            if self.isRandScaleSpecified &&...
                    (~isequal(self.RandXScale,[1 1]) || ~isequal(self.RandYScale,[1 1]))
                error(message('nnet_cnn:imageDataAugmenter:randScalePreferred'))
            end
            
        end % end of constructor/initializer
    end % end of constructor/initializer methods
    
    methods
        % set methods for input validation
        function set.Subcrops(self,subcrops)
            
            validateattributes(subcrops,{'numeric','logical'},{'real','scalar','nonempty'},mfilename,'RandXReflection');
            self.Subcrops = logical(subcrops);
        end
         
    end % end of input validation
end

function fillVal = manageFillValue(A,fillValIn)

if (size(A,3) > 1) && isscalar(fillValIn)
    fillVal = repmat(fillValIn,[1 size(A,3)]);
else
    fillVal = fillValIn;
end

end

function val = selectUniformRandValueFromRange(inputVal,s,propName)

if isa(inputVal,'function_handle')
    val = inputVal();
    switch propName
        case {'RandXShear', 'RandYShear'}
            validateattributes(val,{'numeric'},{'real','finite','scalar','>=',-90,'<=',90},mfilename,propName);
        case {'RandScale', 'RandXScale', 'RandYScale'}
            validateattributes(val,{'numeric'},{'real','finite','scalar','positive'},mfilename,propName);
        case {'RandRotation','RandXTranslation','RandYTranslation'}
            validateattributes(val,{'numeric'},{'real','finite','scalar'},mfilename,propName);
        otherwise
            assert(false,'Invalid Name-Value pair');
    end
else
    val = diff(inputVal) * rand(s) + inputVal(1);
end

end

function validateInput(inputVal,propName)

if ~isa(inputVal,'function_handle')
    
    switch propName
        case {'RandXShear', 'RandYShear'}
            validateattributes(inputVal,{'numeric'},{'real','finite','vector','numel',2,'>=',-90,'<=',90},mfilename,propName);
        case {'RandScale', 'RandXScale', 'RandYScale'}
            validateattributes(inputVal,{'numeric'},{'real','finite','vector','numel',2,'positive'},mfilename,propName);
        case {'RandRotation','RandXTranslation','RandYTranslation'}
            validateattributes(inputVal,{'numeric'},{'real','finite','vector','numel',2},mfilename,propName);
        otherwise
            assert(false,'Invalid Name-Value pair');
    end
    
    if inputVal(1) > inputVal(2)
        error(message('nnet_cnn:imageDataAugmenter:invalidRange',propName));
    end
end
end