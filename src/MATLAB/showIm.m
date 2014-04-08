%% print out image
function showIm(image,label)
figure
imagesc(max(image,[],3))

% set(gcf,'Units','normalized');
% set(gcf,'Position',[0 0 1 1]);

colormap gray

title(label)
axis image
end
