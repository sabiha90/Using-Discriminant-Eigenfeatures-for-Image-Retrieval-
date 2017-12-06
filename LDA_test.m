path_female = dir('/Users/sabihabarlaskar/Documents/MATLAB/FaceClassification_Data/Female/small_*.TIF');
path_male = dir('/Users/sabihabarlaskar/Documents/MATLAB/FaceClassification_Data/Male/small_*.TIF');

tau_female = ones(1024,54);
for i =1:54
    filename_f = strcat('/Users/sabihabarlaskar/Documents/MATLAB/FaceClassification_Data/Female/',path_female(i).name);
    %Read the images %
    I_f = double(imread(filename_f));
    %Vectorize the images into a column vector of size N^2 X 1 and store it
    %in a matrix to form an image space
    V_I_f = I_f(:);
    tau_female(:,i) = V_I_f;
end
mean_female = mean(tau_female,2);

tau_male = ones(1024,45);
for i = 1:45
    filename_m = strcat('/Users/sabihabarlaskar/Documents/MATLAB/FaceClassification_Data/Male/',path_male(i).name);
    %Read the images %
    I_m = double(imread(filename_m));
    %Vectorize the images into a column vector of size N^2 X 1 and store it
    %in a matrix to form an image space
    V_I_m = I_m(:);
    tau_male(:,i) = V_I_m;
end
%mean_male = mean(tau_male,2);
tau_all = [tau_female tau_male];
psi = mean(tau_all,2);

phi = tau_all - repmat(psi,1,99);
%A is a matrix that contains the phi of all the images
A = phi;
C = A * A';
[eigvec,eigval] = eig(C);
eigval = diag(eigval);
[sortedeigval, eig_indices] = sort(abs(eigval),'descend');
Sorted_eig = eigvec(:,eig_indices);
size(sortedeigval);
total_eigval = sum(sortedeigval);
var_covered = sortedeigval(1);
k = 2;
while((var_covered/total_eigval)<0.95)
    var_covered = var_covered + sortedeigval(k);
    k = k + 1;
end
top_k_eig_vec = Sorted_eig(:,1:k);
size(top_k_eig_vec);
% figure('NumberTitle','off','Name','Eigen faces')
% 
% for i = 1:k
%     colormap('gray');
%     subplot(10,5,i);
%     imagesc(reshape(top_k_eig_vec(:, i), 32, 32)); 
% end
projected_tau_all = top_k_eig_vec' * phi;
projected_tau_female = projected_tau_all(:,1:54);
projected_tau_male = projected_tau_all(:,55:99);

mean_all = mean(projected_tau_all,2);
mean_female= mean(projected_tau_female,2);
mean_male= mean(projected_tau_male,2);



diff_female = projected_tau_female - repmat(mean_female,1,54);
diff_male = projected_tau_male - repmat(mean_male,1,45);

S_female = diff_female * diff_female';
S_male = diff_male * diff_male';

S_w = S_female + S_male;

diff_female_from_mean = mean_female - mean_all;
diff_male_from_mean = mean_male - mean_all;

S_female_from_mean = diff_female_from_mean * diff_female_from_mean';
S_male_from_mean = diff_male_from_mean * diff_male_from_mean';

S_b = S_female_from_mean + S_male_from_mean;

S_mat = (inv(S_w))*(S_b);

[eigvec_lda,eigval_lda] = eig(S_mat);
eigval_lda = diag(eigval_lda);
[sortedeigval_lda, eig_indices_lda] = sort(abs(eigval_lda),'descend');
