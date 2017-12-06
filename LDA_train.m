path_train_female = dir('/Users/sabihabarlaskar/Documents/MATLAB/LDA_data/Train_Female/small_*.TIF');
path_train_male = dir('/Users/sabihabarlaskar/Documents/MATLAB/LDA_data/Train_Male/small_*.TIF');
[female_n,female_col] = size(path_train_female);
tau_female = ones(1024,female_n);

for i =1:female_n
    filename_f = strcat('/Users/sabihabarlaskar/Documents/MATLAB/LDA_data/Train_Female/',path_train_female(i).name);
    %Read the images %
    I_f = double(imread(filename_f));
    %Vectorize the images into a column vector of size N^2 X 1 and store it
    %in a matrix to form an image space
    V_I_f = I_f(:);
    tau_female(:,i) = V_I_f;
end

[male_n,male_col] = size(path_train_male);

tau_male = ones(1024,male_n);
for i = 1:male_n
    filename_m = strcat('/Users/sabihabarlaskar/Documents/MATLAB/LDA_data/Train_Male/',path_train_male(i).name);
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
n = female_n + male_n;
phi = tau_all - repmat(psi,1,n);
size(phi);
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
while((var_covered/total_eigval)<0.93)
    var_covered = var_covered + sortedeigval(k);
    k = k + 1;
end
k
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
projected_tau_female = projected_tau_all(:,1:female_n);
projected_tau_male = projected_tau_all(:,female_n+1:n);
%scatter(projected_tau_female,projected_tau_male);
size(projected_tau_male);
mean_all = mean(projected_tau_all,2);
mean_female= mean(projected_tau_female,2);
mean_male= mean(projected_tau_male,2);



diff_female = projected_tau_female - repmat(mean_female,1,female_n);
size(diff_female);
diff_male = projected_tau_male - repmat(mean_male,1,male_n);
%Step 6
S_b_female = diff_female * diff_female';
S_b_male = diff_male * diff_male';
%Step 6 
S_w = S_b_female + S_b_male;

diff_female_from_mean = mean_female - mean_all;
diff_male_from_mean = mean_male - mean_all;
%Covariance of LDA
S_female_from_mean = diff_female_from_mean * diff_female_from_mean';
S_male_from_mean = diff_male_from_mean * diff_male_from_mean';

S_b = S_female_from_mean + S_male_from_mean;

S_mat = (inv(S_w))*(S_b);

[eigvec_lda,eigval_lda] = eig(S_mat);
eigval_lda = diag(eigval_lda);
[sortedeigval_lda, eig_indices_lda] = sort(abs(eigval_lda),'descend');

Sorted_eig_lda = eigvec_lda(:,eig_indices_lda);
principle_comp = Sorted_eig_lda(:,1);

discriminant_slope = principle_comp' * top_k_eig_vec';
discriminant_intercept = principle_comp' * ((mean_female + mean_male)/2);

size(discriminant_slope);
size(discriminant_intercept);
test_condition = discriminant_slope * (tau_male - psi) - discriminant_intercept;
%male(negative)
%female(positive)

%Testing

path_test_female = dir('/Users/sabihabarlaskar/Documents/MATLAB/LDA_data/Test_female/small_*.TIF');
path_test_male = dir('/Users/sabihabarlaskar/Documents/MATLAB/LDA_data/Test_male/small_*.TIF');
[female_t_n,t_col] = size(path_test_female);
tau_female_test = ones(1,female_t_n);
for i =1:female_t_n
    filename_f_test = strcat('/Users/sabihabarlaskar/Documents/MATLAB/LDA_data/Test_female/',path_test_female(i).name);
    %Read the test images %
    I_f_test = double(imread(filename_f_test));
    V_I_f_test = I_f_test(:);
    %tau_female_test(:,i) = V_I_f_test;
    test_condition_female = discriminant_slope * (V_I_f_test - psi) - discriminant_intercept;
    tau_female_test(i) = test_condition_female;
    
end

[male_t_n,t_m_col] = size(path_test_male);
tau_male_test = ones(1,male_t_n);
for i =1:male_t_n
    filename_m_test = strcat('/Users/sabihabarlaskar/Documents/MATLAB/LDA_data/Test_male/',path_test_male(i).name);
    %Read the test images %
    I_m_test = double(imread(filename_m_test));
    V_I_m_test = I_m_test(:);
    %tau_male_test(:,i) = V_I_m_test;
    test_condition_male = discriminant_slope * (V_I_m_test - psi) - discriminant_intercept;
    tau_male_test(i) = test_condition_male;
end


count1 = 0;
for i = 1:8
    if tau_male_test(i)<0
        count1 = count1+1;
    end
end
acc_male = count1/8

count2 = 0;
for i = 1:10
    if tau_female_test(i) > 0
        count2 =  count2+1;
    end
end

acc_female = count2/10

%test_condition = discriminant_slope * (V_I_f_test - psi) - discriminant_intercept