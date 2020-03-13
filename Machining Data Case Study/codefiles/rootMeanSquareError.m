function rms_err = rootMeanSquareError(pred,tgt)
    err = (pred-tgt).^2;
    msqerr = mean(err);
    rms_err = sqrt(msqerr);
end