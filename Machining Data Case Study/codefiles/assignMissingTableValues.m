
aftable = allFiles_Table;
varlist = {'vc','fd','ae','rrc','ylabel','t','wearState','Imputed'};
for varname = 1:numel(varlist)
    if strcmp(varlist(varname),'wearState')
            aftable = addvars(aftable,zeros(height(aftable),1),'After',aftable.Properties.VariableNames(end),'NewVariableNames',varlist(varname));
            aftable.wearState(:) = "e";
    else
    aftable = addvars(aftable,zeros(height(aftable),1),'After',aftable.Properties.VariableNames(end),'NewVariableNames',varlist(varname));
    end
end
%%
aftable.Properties.VariableNames([3,4])={'exp','xlabel'};
aftable = movevars(aftable,'xlabel','after','Variant');
%%
aftable_joined = innerjoin(aftable,cdata_yest);



for r = 1:height(aftable)
    % find the row in ctd_yest which matches the xlabel and experiment
    matches_in_cdata = cdata_yest.xlabel == aftable.xlabel(r) & cdata_yest.exp == aftable.exp(r);
    disp('Found matches: '+numel(matches_in_cdata))
    aftable(r,varlist) = cdata_yest(matches_in_cdata,varlist);
end
