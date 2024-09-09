% Streamer data visualization script
% Usage: run the script, select a streamer output file and it will be
% plotted automatically. You can also run a section at a time so that
% you can re-plot without re-selecting a CSV every time

%% Read CSV

[csvname, csvpath] = uigetfile("*.csv", "Select CSV");
if (isnumeric(csvname) || isnumeric(csvpath))
    disp("User did not select a file")
    return
end

%% Plot

% set the corresponding array element in plotMask to 1 to plot that channel
% set to 0 to not plot it.
% i.e. to plot channels 1 and 3, set plotMask to: [1 0 1 0 0 0 0 0]
plotMask = [1 1 1 1 1 1 1 1];

filepath = append(csvpath, csvname);
dataTable = readtable(append(csvpath, csvname));

rawData = table2array(dataTable);
data = rawData(:,2:end);
timestamps = zeros(size(data)) + rawData(:,1);

plot(timestamps(:, plotMask == 1), data(:, plotMask == 1));
allNames = dataTable.Properties.VariableNames(2:end);
legend(allNames(plotMask == 1));
title('Streamer Data');
xlabel('Time (s)');
ylabel('Value');
