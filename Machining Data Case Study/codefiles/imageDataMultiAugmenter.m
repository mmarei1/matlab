classdef imageDataMultiAugmenter < handle
    % imageDataAugmenter Configure image data augmentation
    %
    %   aug = imageDataAugmenter() creates an imageDataAugmenter object
    %   with default property values. The default state of the
    %   imageDataAugmenter is the identity transformation.
    %
    %   aug = imageDataAugmenter(Name,Value,___) configures a set of image
    %   augmentation options using Name/Value pairs to set properties.
    %
    %   imageDataAugmenter properties:
    %       FillValue           - Value used to define out of bounds points
    %       RandXReflection     - Random X reflection
    %       RandYReflection     - Random Y reflection
    %       RandRotation        - Random rotation
    %       RandScale           - Random uniform scale in X & Y direction
    %       RandXScale          - Random X scale
    %       RandYScale          - Random Y scale
    %       RandXShear          - Random X shear
    %       RandYShear          - Random Y shear
    %       RandXTranslation    - Random X translation
    %       RandYTranslation    - Random Y translation
    %
    %   imageDataAugmenter methods:
    %       augment                - Return augmented images
    %
    %
    %   Example 1
    %   ---------
    %   Train a convolutional neural network on some synthetic images of
    %   handwritten digits. Apply random rotations during training to add
    %   rotation invariance to trained network.
    %
    %   [XTrain, YTrain] = digitTrain4DArrayData;
    %
    %   imageSize = [28 28 1];
    %
    %   layers = [ ...
    %       imageInputLayer(imageSize,'Normalization','none');
    %       convolution2dLayer(5,20);
    %       reluLayer();
    %       maxPooling2dLayer(2,'Stride',2);
    %       fullyConnectedLayer(10);
    %       softmaxLayer();
    %       classificationLayer()];
    %
    %   opts = trainingOptions('sgdm','Plots','training-progress');
    %
    %   imageAugmenter = imageDataAugmenter('RandRotation',[-10 10]);
    %
    %   datasource = augmentedImageSource(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);
    %
    %   net = trainNetwork(datasource,layers,opts);
    %
    %   Example 2
    %   ---------
    %   Use function handle to specify values for the properties and view
    %   augmented images
    %
    %   % Select a RandRotation value between the interval [5,15]
    %   imageAugmenter = imageDataAugmenter('RandRotation',@() 5 + (15-5)*rand);
    %
    %   % Augment multiple images with identical augmentations
    %   img1 = imread('peppers.png');
    %   img2 = imread('ngc6543a.jpg');
    %   outCellArray = augment(imageAugmenter,{img1, img2});
    %
    %   % View augmented images
    %   outImg = imtile(outCellArray);
    %   imshow(outImg);
    %
    % See also augmentedImageDatastore, imageInputLayer, trainNetwork

    % Copyright 2017-2018 The MathWorks, Inc.
    
    properties (SetAccess = 'private')
        
        %FillValue - Value used to define out of bounds points when resampling
        %
        %    FillValue is a numeric scalar or vector. If images being
        %    augmented are single channel, FillValue must be scalar. If
        %    images being augmented are multichannel then FillValue may be
        %    a scalar or a vector with length equal to the number of
        %    channels in the input image data. The default fill value for
        %    categorical images is an '<undefined>' label.
        FillValue
        
        %RandXReflection - Random X reflection
        %
        %    RandXReflection is a logical scalar that specifies whether
        %    random left/right reflections are applied to input image data.
        %
        %    Default: false
        %
        RandXReflection
        
        %RandYReflection - Random Y reflection
        %
        %    RandYReflection is a logical scalar that specifies whether
        %    random up/down reflections are applied to input image data.
        %
        %    Default: false
        %
        RandYReflection
        
        %RandRotation - Random rotation
        %
        %    RandRotation is a two element numeric vector that specifies
        %    the random range. RandRotation can also be a function handle
        %    that takes no input arguments and returns a real scalar. The
        %    values are in degress, of rotations that are applied to input
        %    image data
        %
        %    Default: [0 0]
        RandRotation
        
        %RandScale - Random uniform scale in X & Y direction
        %
        %    RandScale is a two element numeric vector that specifies the
        %    random range. RandScale can also be a function handle that
        %    takes no input arguments and returns a positive real scalar.
        %    The values specify the scale that is applied to input image
        %    data in X and Y dimensions uniformly.
        %
        %    Default: [1 1]
        RandScale
        
        %RandXScale - Random X scale
        %
        %    RandXScale is a two element numeric vector that specifies the
        %    random range. RandXScale can also be a function handle that
        %    takes no input arguments and returns a positive real scalar.
        %    The values specify the scale that is applied to input image
        %    data in X dimension. RandXScale is ignored when RandScale is
        %    provided.
        %
        %    Default: [1 1]
        RandXScale
        
        %RandYScale - Random Y scale
        %
        %    RandYScale is a two element numeric vector that specifies the
        %    random range. RandYScale can also be a function handle that
        %    takes no input arguments and returns a positive real scalar.
        %    The values specify the scale that is applied to input image
        %    data in Y dimension. RandYScale is ignored when RandScale is
        %    provided.
        %
        %    Default: [1 1]
        RandYScale
        
        %RandXShear - Random X shear
        %
        %    RandXShear is a two element numeric vector that specifies the
        %    random range. RandXShear can also be a function handle that
        %    takes no input arguments and returns a real scalar. The values
        %    specify shear in the X dimension that is applied to input
        %    image data. Shear is specified in terms of shear angles
        %    measured in units of degres. The valid range of shear angles
        %    is (-90,90) degrees.
        %
        %    Default: [0 0]
        RandXShear
        
        %RandYShear - Random Y shear
        %
        %    RandYShear is a two element numeric vector that specifies the
        %    random range. RandYShear can also be a function handle that
        %    takes no input arguments and returns a real scalar. The values
        %    specify shear in the Y dimension that is applied to input
        %    image data. Shear is specified in terms of shear angles
        %    measured in units of degres. The valid range of shear angles
        %    is (-90,90) degrees.
        %
        %    Default: [0 0]
        RandYShear
        
        %RandXTranslation - Random X translation
        %
        %    RandXTranslation is a two element numeric vector that
        %    specifies the random range. RandXTranslation can also be a
        %    function handle that takes no input arguments and returns a
        %    real scalar. The values specify translation in the X
        %    dimension, in units of pixels, that is applied to input image
        %    data.
        %
        %    Default: [0 0]
        RandXTranslation
        
        %RandYTranslation - Random Y translation
        %
        %    RandYTranslation is a two element numeric vector that
        %    specifies the random range. RandYTranslation can also be a
        %    function handle that takes no input arguments and returns a
        %    real scalar. The values specify translation in the Y
        %    dimension, in units of pixels, that is applied to input image
        %    data.
        %
        %    Default: [0 0]
        RandYTranslation
        
        %RandSubcrops
        %
        %   RandSubcrops is a scalar integer specifying the number of
        %   random image subcrops to break the image into. If the number of
        %   subcrops is already specified, this method does not resolve.
        %
        %   Default: 0
        RandSubcrops        
    end
    
    % Public, cache intermediate state of transform to allow for testing
    properties (Hidden)
        
        Rotation
        XReflection
        YReflection
        Scale
        XScale
        YScale
        XShear
        YShear
        XTranslation
        YTranslation
        Subcrops
        
        isRandScaleSpecified
        AffineTransforms
        
    end
    
    properties (Hidden)
        % RandStream used for random number generation
        Stream
    end
    
    methods
        
        function self = imageDataMultiAugmenter(varargin)
            %imageDataAugmenter Construct imageDataAugmenter object.
            %
            %   augmenter = imageDataAugmenter() constructs an
            %   imageDataAugmenter object with default property settings.
            %
            %   augmenter = imageDataAugmenter('Name',Value,___) specifies
            %   parameters that control aspects of data augmentation.
            %   Parameter names can be abbreviated and case does not
            %   matter.
            %
            %   Parameters include:
            %
            %   'FillValue'         A numeric scalar or vector 
            %                       (when augmenting multi-channel images)
            %                       that defines the value used during
            %                       resampling when transformed points fall
            %                       out of bounds. The default fill value
            %                       for categorical images is an
            %                       '<undefined>' label.
            %
            %                       Default: 0
            %
            %   'RandXReflection'   A scalar logical that defines whether
            %                       random left/right reflections are
            %                       applied.
            %
            %                       Default: false
            %
            %   'RandYReflection'   A scalar logical that defines whether
            %                       random up/down reflections are applied.
            %
            %                       Default: false
            %
            %   'RandRotation'      A two element vector that defines the
            %                       uniform range or a function handle
            %                       returning a scalar, in units of
            %                       degrees, of rotations that will be
            %                       applied.
            %
            %                       Default: [0 0]
            %
            %   'RandScale'         A two element vector that defines the
            %                       uniform range or a function handle
            %                       returning a scalar. The values define
            %                       the scale that will applied in the X
            %                       and Y dimension uniformly.
            %
            %                       Default: [1 1]
            %
            %   'RandXScale'        A two element vector that defines the
            %                       uniform range or a function handle
            %                       returning a scalar. The values define
            %                       the scale that will applied in the X
            %                       dimension. RandXScale is ignored when
            %                       RandScale is provided.
            %
            %                       Default: [1 1]
            %
            %   'RandYScale'        A two element vector that defines the
            %                       uniform range or a function handle
            %                       returning a scalar. The values define
            %                       the scale that will applied in the Y
            %                       dimension. RandYScale is ignored when
            %                       RandScale is provided.
            %
            %                       Default: [1 1]
            %
            %   'RandXShear'        A two element vector that defines the
            %                       uniform range or a function handle
            %                       returning a scalar. The values define
            %                       the shear that will be applied in the X
            %                       dimension. Specified as a shear angle
            %                       in units of degrees.
            %
            %                       Default: [0 0]
            %
            %   'RandYShear'        A two element vector that defines the
            %                       uniform range or a function handle
            %                       returning a scalar. The values define
            %                       the shear that will be applied in the Y
            %                       dimension. Specified as a shear angle
            %                       in units of degrees.
            %
            %                       Default: [0 0]
            %
            %   'RandXTranslation'  A two element vector that defines the
            %                       uniform range or a function handle
            %                       returning a scalar. The values specify
            %                       translation that will be applied in the
            %                       X dimension. Specified in units of
            %                       pixels.
            %
            %                       Default: [0 0]
            %
            %   'RandYTranslation'  A two element vector that defines the
            %                       uniform range or a function handle
            %                       returning a scalar. The values specify
            %                       translation that will be applied in the
            %                       Y dimension. Specified in units of
            %                       pixels.
            %
            %                       Default: [0 0]
            % 
            %   'Subcrops'          A scalar value that specifies the
            %                       number of image subcrops to extract
            %                       from the image. 
            %                       
            %                       Default: 0
            %
            %   'Size'              A scalar value that specifies the pixel
            %                       size of each output subcrop.
            % 
            %                       Default: 250
            %
            self.parseInputs(varargin{:});
            self.Stream = RandStream('mt19937ar','Seed',0);
            
            self.isRandScaleSpecified = false;
            if ~isequal(self.RandScale,[1 1])
                self.isRandScaleSpecified = true;
            end
            
            self.isNumSubcropsSpecified = false;
            if ~isequal(self.Subcrops,0)
                self.isRandScaleSpecified = true;
            end
            
            %Error if RandXScale or RandYScale specified along with
            %RandScale
            if self.isRandScaleSpecified &&...
                    (~isequal(self.RandXScale,[1 1]) || ~isequal(self.RandYScale,[1 1]))
                error(message('nnet_cnn:imageDataAugmenter:randScalePreferred'))
            end
            
        end
        
    end
    
    % set methods for input validation
    methods
        
        function set.RandSubcrops(self,numSubcrops)
            
            validateattributes(numSubcrops,{'numeric','logical'},{'real','scalar','nonempty'},mfilename,'RandXReflection');
            self.RandSubcrops = logical(numSubcrops);
            
        end
        
        function set.RandXReflection(self,xReflect)
            
            validateattributes(xReflect,{'numeric','logical'},{'real','scalar','nonempty'},mfilename,'RandXReflection');
            self.RandXReflection = logical(xReflect);
            
        end
        
        function set.RandYReflection(self,yReflect)
            
            validateattributes(yReflect,{'numeric','logical'},{'real','scalar','nonempty'},mfilename,'RandYReflection');
            self.RandYReflection = logical(yReflect);
            
        end
        
        function set.RandRotation(self,rotationInDegrees)
            
            validateInput(rotationInDegrees,'RandRotation');
            self.RandRotation = rotationInDegrees;
            
        end
        
        function set.RandScale(self,Scale)
            
            validateInput(Scale,'RandXScale');
            self.RandScale = Scale;
            
        end
        
        function set.RandXScale(self,xScale)
            
            validateInput(xScale,'RandXScale');
            self.RandXScale = xScale;
            
        end
        
        function set.RandYScale(self,yScale)
            
            validateInput(yScale,'RandYScale');
            self.RandYScale = yScale;
            
        end
        
        function set.RandXShear(self,xShear)
            
            validateInput(xShear,'RandXShear');
            self.RandXShear = xShear;
            
        end
        
        function set.RandYShear(self,yShear)
            
            validateInput(yShear,'RandYShear');
            self.RandYShear = yShear;
            
        end
        
        function set.RandXTranslation(self,xTrans)
            
            validateInput(xTrans,'RandXTranslation');
            self.RandXTranslation = xTrans;
            
        end
        
        function set.RandYTranslation(self,yTrans)
            
            validateInput(yTrans,'RandYTranslation');
            self.RandYTranslation = yTrans;
            
        end
                
        function set.FillValue(self,fillIn)
            validateattributes(fillIn,{'numeric'},{'real','vector'},mfilename,'FillValue');
            
            if (length(fillIn) ~= 1) && (length(fillIn) ~= 3)
                error(message('nnet_cnn:imageDataAugmenter:invalidFillValue'));
            end
            
            self.FillValue = fillIn;
        end
        
    end
    
    methods (Access = 'private')
        
        function selectRandValues(self)
            
            if self.RandXReflection
                xReflect = sign(rand(self.Stream) - 0.5);
            else
                xReflect = 1;
            end
            
            if self.RandYReflection
                yReflect = sign(rand(self.Stream) - 0.5);
            else
                yReflect = 1;
            end
            
            self.XReflection = xReflect;
            self.YReflection = yReflect;
            

            self.Rotation = selectUniformRandValueFromRange(self.RandRotation,self.Stream,'RandRotation');
            self.XShear = selectUniformRandValueFromRange(self.RandXShear,self.Stream,'RandXShear');
            self.YShear = selectUniformRandValueFromRange(self.RandYShear,self.Stream,'RandYShear');
            self.XTranslation = selectUniformRandValueFromRange(self.RandXTranslation,self.Stream,'RandXTranslation');
            self.YTranslation = selectUniformRandValueFromRange(self.RandYTranslation,self.Stream,'RandYTranslation');
            
            % If RandScale is specified, XScale and YScale should be equal
            if self.isRandScaleSpecified
                self.Scale = selectUniformRandValueFromRange(self.RandScale,self.Stream,'RandScale');
                self.XScale = self.Scale;
                self.YScale = self.Scale;
            else
                self.XScale = selectUniformRandValueFromRange(self.RandXScale,self.Stream,'RandXScale');
                self.YScale = selectUniformRandValueFromRange(self.RandYScale,self.Stream,'RandYScale');
            end
        end
        
        function Tout = makeAffineTransform(self,inputSize)
            
            centerXShift = (inputSize(2)-1)/2;
            centerYShift = (inputSize(1)-1)/2;
            
            [moveToOriginTransform,moveBackTransform] = deal(eye(3));
            moveToOriginTransform(3,1) = -centerXShift;
            moveToOriginTransform(3,2) = -centerYShift;
            moveBackTransform(3,1) = centerXShift;
            moveBackTransform(3,2) = centerYShift;
             
            centeredRotation = moveToOriginTransform * self.makeRotationTransform() * moveBackTransform;
            centeredShear = moveToOriginTransform * self.makeShearTransform() * moveBackTransform;
            centeredReflection = moveToOriginTransform * self.makeReflectionTransform() * moveBackTransform;
            centeredScale = moveToOriginTransform * self.makeScaleTransform() * moveBackTransform;
            
            Tout = centeredRotation * centeredShear * centeredScale * centeredReflection * self.makeTranslationTransform();
              
        end
        
        function parseInputs(self,varargin)
            
            p = inputParser();
            p.addParameter('FillValue',0);
            p.addParameter('RandXReflection',false);
            p.addParameter('RandYReflection',false);
            p.addParameter('RandRotation',[0 0]);
            p.addParameter('RandScale', [1, 1]);
            p.addParameter('RandXScale',[1, 1]);
            p.addParameter('RandYScale',[1, 1]);
            p.addParameter('RandXShear',[0, 0]);
            p.addParameter('RandYShear',[0, 0]);
            p.addParameter('RandXTranslation',[0,0]);
            p.addParameter('RandYTranslation',[0,0]);
            p.addParameter('RandNumSubcrops',0);
            p.parse(varargin{:});
            params = p.Results;
            
            self.FillValue = params.FillValue;
            self.RandXReflection = params.RandXReflection;
            self.RandYReflection = params.RandYReflection;
            self.RandRotation = params.RandRotation;
            self.RandScale = params.RandScale;
            self.RandXScale = params.RandXScale;
            self.RandYScale = params.RandYScale;
            self.RandXShear = params.RandXShear;
            self.RandYShear = params.RandYShear;
            self.RandXTranslation = params.RandXTranslation;
            self.RandYTranslation = params.RandYTranslation;
            self.RandNumSubcrops = params.RandNumSubcrops;
        end
        
        function tform = makeRotationTransform(self)
                        
            tform = [cosd(self.Rotation), -sind(self.Rotation), 0;...
                sind(self.Rotation), cosd(self.Rotation), 0;...
                0, 0, 1];
            
        end
        
        function tform = makeShearTransform(self)
            
            tform = [1, tand(self.YShear), 0;...
                tand(self.XShear), 1, 0;...
                0, 0, 1];
            
        end
        
        function tform = makeScaleTransform(self)
            
            tform = [self.XScale, 0, 0;...
                0, self.YScale, 0;...
                0, 0, 1];
            
        end
        
        function tform = makeReflectionTransform(self)
            
            tform = [self.XReflection, 0, 0;...
                0, self.YReflection, 0;...
                0, 0, 1];
            
        end
        
        function tform = makeTranslationTransform(self)
           tform = eye(3);
           tform(3,1) = self.XTranslation;
           tform(3,2) = self.YTranslation;
        end
        
    end
    
    methods 
        
        function B = augment(self,A)
            %augment Augment input image data.
            %
            %   augmentedImage = augment(augmenter,A) performs image
            %   augmentation on the input A. A can be a numeric image or a
            %   cell array of numeric and categorical images. When A is a cell array of
            %   images, the augment function performs identical
            %   augmentations on all the images and returns a cell array B
            %   of augmented images. Images in a cell array can be of
            %   different sizes and types.
                
            if iscell(A)    
                B = cell(size(A));
                self.AffineTransforms = zeros(3,3,length(A)); % 3 x 3 x batchSize matrix
                
                % Select Rand values before the for loop so that same
                % augmentations are applied to all the images
                self.selectRandValues();
                
                for img = 1:numel(A)
                    if (isnumeric(A{img}) || iscategorical(A{img})) && (ndims(A{img}) < 4)
                        
                        interp = 'linear';
                        imgIsCategorical = false;
                        fillValue = manageFillValue(A{img},self.FillValue);
                        
                        if(iscategorical(A{img}))
                            % Convert image to numeric for warp and save
                            % current categories for converting the result
                            % back to categorical
                            imgIsCategorical = true;
                            interp = 'nearest';
                            cats = categories(A{img});
                            fillValue = str2double(cats{end}) + 1;
                            A{img} = double(A{img});
                        end
                        
                        % tform could be different for different images as
                        % centered rotation is performed and that depends
                        % on size of the image
                        tform = self.makeAffineTransform(size(A{img}));
                        
                        B{img} = nnet.internal.cnnhost.warpImage2D(A{img},tform(1:3,1:2),interp,fillValue);
                        self.AffineTransforms(:,:,img) = tform;
                        
                        if imgIsCategorical
                            % If original image was categorical, covert the
                            % result to categorical
                            B{img} = categorical(B{img}, 1:numel(cats), cats);
                        end
                    else
                        error(message('nnet_cnn:imageDataAugmenter:invalidImageCellarray',img));
                    end
                end
                
            elseif isnumeric(A) && (ndims(A) < 4)
                self.selectRandValues();
                tform = self.makeAffineTransform(size(A));
                B = nnet.internal.cnnhost.warpImage2D(A,tform(1:3,1:2),'linear',manageFillValue(A,self.FillValue));
                self.AffineTransforms = tform;
            else
                error(message('nnet_cnn:imageDataAugmenter:invalidImage'));
            end
            
        end
    end
    
    methods(Hidden)
        
        function [A, B] = augmentPair(self, X, Y, interpY, fillValueY)
            %augmentPair Augment inputs X and Y.
            %
            %   [A, B] = augmentPair(augmenter, X, Y) performs the same
            %   image augmentation on the input image X and Y. X and Y are
            %   MxNxC matrices or B element cell arrays containing MxNxC
            %   iamges.
            %
            %   [...] = augmentPair(augmenter, X, Y, interpY, fillValueY)
            %   optionally specify the interpolation method and fill value
            %   to use for augmenting Y. By default, interpY is 'linear'
            %   and fillValueY is augmenter.FillValue.
                                 
            if nargin < 5
                fillValueY = self.FillValue;
            end
            
            if nargin < 4
                interpY = 'linear';                
            end
                        
            A = self.augment(X);
            tform = self.AffineTransforms;            
            B = self.augmentY(Y, tform, interpY, fillValueY);                        
            
        end
        
        function B = augmentY(~, Y, tform, interp, fillValue)
            % Augment using known transform. fillValue can be a scalar,
            % M-by-3, or column vector of length(Y), when Y is a cell array. 
            
            if iscell(Y)
                
                if isscalar(fillValue) || (isrow(fillValue) && numel(fillValue) == 3)
                    fillValue = repelem(fillValue,numel(Y),1);
                end
                
                B = cell(size(Y));
                for img = 1:numel(Y)   
                    fv = manageFillValue(Y{img},fillValue(img,:));
                    B{img} = nnet.internal.cnnhost.warpImage2D(Y{img},tform(1:3,1:2,img),interp,fv);
                end
                
            elseif isnumeric(Y) && (ndims(Y) < 4)               
                B = nnet.internal.cnnhost.warpImage2D(Y,tform(1:3,1:2),interp, manageFillValue(Y,fillValue));
            else
                error(message('nnet_cnn:imageDataAugmenter:invalidImage'));
            end
        end
        
    end
    
    methods(Static, Hidden = true)
        function self = loadobj(S)
            % RandScale property was introduced in R2018b
            if isfield(S,'RandScale')
                self = imageDataAugmenter('FillValue',S.FillValue,...
                'RandXReflection',S.RandXReflection,...
                'RandYReflection',S.RandYReflection,...
                'RandRotation',S.RandRotation,...
                'RandScale',S.RandScale,...
                'RandXScale',S.RandXScale,...
                'RandYScale',S.RandYScale,...
                'RandXShear',S.RandXShear,...
                'RandYShear',S.RandYShear,...
                'RandXTranslation',S.RandXTranslation,...
                'RandYTranslation',S.RandYTranslation);
                
            else
                self = imageDataAugmenter('FillValue',S.FillValue,...
                    'RandXReflection',S.RandXReflection,...
                    'RandYReflection',S.RandYReflection,...
                    'RandRotation',S.RandRotation,...
                    'RandXScale',S.RandXScale,...
                    'RandYScale',S.RandYScale,...
                    'RandXShear',S.RandXShear,...
                    'RandYShear',S.RandYShear,...
                    'RandXTranslation',S.RandXTranslation,...
                    'RandYTranslation',S.RandYTranslation);
            end
            
            if isfield(S,'Stream')
                self.Stream = S.Stream; 
            end
                
        end
    end
    
    methods (Hidden)
        function S = saveobj(self)
            % Serialize denoisingImageDatasource object
            S = struct('FillValue',self.FillValue,...
                'RandXReflection',self.RandXReflection,...
                'RandYReflection',self.RandYReflection,...
                'RandRotation',self.RandRotation,...
                'RandScale',self.RandScale,...
                'RandXScale',self.RandXScale,...
                'RandYScale',self.RandYScale,...
                'RandXShear',self.RandXShear,...
                'RandYShear',self.RandYShear,...
                'RandXTranslation',self.RandXTranslation,...
                'RandYTranslation',self.RandYTranslation);
            
           if isprop(self,'Stream')
               S.Stream = self.Stream;
           end
        end
        
    end
       
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
        case {'NumSubcrops'}
            validateattributes(val,{'numeric'},{'real','finite','scalar','numel',1},mfilename,propName);
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
        case {'NumSubcrops'}
            validateattributes(inputVal,{'numeric'},{'real','finite','scalar','numel',1},mfilename,propName);
        otherwise
            assert(false,'Invalid Name-Value pair');
    end
    
    if inputVal(1) > inputVal(2)
        error(message('nnet_cnn:imageDataAugmenter:invalidRange',propName));
    end
end
end