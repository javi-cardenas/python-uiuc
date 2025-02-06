import numpy as np


class Question1(object):
    def rotate_matrix(self, theta):
        # convert theta from degrees to radians
        theta *= (np.pi/180)
        
        # rotation matrix from problem 1
        R_theta = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]])
        return R_theta

    def rotate_2d(self, points, theta):
        rot_points = self.rotate_matrix(theta) @ points # the dot product of the given points and rotation matrix
        return rot_points

    def combine_rotation(self, theta1, theta2):
        combined_rotation_matrix = self.rotate_matrix(theta1 + theta2) # the combined rotation matrix for 2 rotations
        
        R          = self.rotate_matrix(theta1)
        R_prime    = self.rotate_matrix(theta2)
        combined_R = R_prime @ R
        
        norm1 = np.linalg.norm(combined_rotation_matrix, 'fro')
        norm2 = np.linalg.norm(combined_R, 'fro')
        err = norm1 - norm2
        
        return err


class Question2(object):
    def rotate_matrix_x(self, theta):
         # convert theta from degrees to radians
        theta *= (np.pi/180)
        
        Rx_matrix = [[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]]
        R_x = np.array(Rx_matrix)
        return R_x

    def rotate_matrix_y(self, theta):
         # convert theta from degrees to radians
        theta *= (np.pi/180)
        
        Ry_matrix = [[np.cos(theta) , 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]]
        R_y = np.array(Ry_matrix)
        return R_y

    def rotate_matrix_z(self, theta):
         # convert theta from degrees to radians
        theta *= (np.pi/180)
        
        Rz_matrix = [[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta) , 0],
                     [0, 0, 1]]
        R_z = np.array(Rz_matrix)
        return R_z

    def rot_matrix(self, alpha, beta, gamma):
        # rotation matrix that rotates points in this sequence: z -> y -> z
        rotated_z = self.rotate_matrix_z(alpha)
        rotated_y = self.rotate_matrix_y(beta)  @ rotated_z
        rotated_z = self.rotate_matrix_z(gamma) @ rotated_y
        
        R = rotated_z
        return R

    def rotate_point(self, points, R):
        rot_points = R @ points
        return rot_points


class Question3(object):
    def rotate_x_axis(self, image_size, theta):
        # points that correspond to the x-axis from -N to N where the image is of size 2N+1 X 2N+1
        points = np.array([(x,0) for x in range(-(image_size-1)//2, (image_size-1)//2 + 1)])
        
        q1 = Question1()
        rot_points = q1.rotate_2d(points.T,theta)
        return rot_points

    def nudft2(self, img, grid_f):
        
        # constants
        N = (len(img)-1)//2
        complex_i = np.complex128(0-1j)
        grid_f = grid_f.T # take the transpose to easily access points

        
        img_f = np.zeros(len(grid_f),dtype=np.complex128)

        for i in range(len(grid_f)):
            # grab (k_x,k_y) pair
            k_x = grid_f[i,0]
            k_y = grid_f[i,1]

            # create the mesh grid
            x = np.linspace(-N, N, len(img))
            y = np.linspace(-N, N, len(img))
            xv, yv = np.meshgrid(x,y)

            # formula
            argument  = complex_i * ((2*np.pi)/len(img)) * (xv*k_x + yv*k_y)
            exp = np.exp(argument)
            ft  = np.sum(np.sum(img*exp))
            img_f[i] = ft
            
        return img_f

    def gen_projection(self, img, theta):
        # grid on which to compute the fourier transform
        points_rot = None

        # Put your code here
        
        # create rotated grid
        points_rot = self.rotate_x_axis(len(img), theta)
        
        # Don't change the rest of the code and the output!
        ft_img = self.nudft2(img, points_rot)
        proj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ft_img)))
        proj = np.real(proj)
        return proj


class Question4(object):
    def nudft3(self, vol, grid_f):
        
        
#         grid_f = np.zeros(9).reshape(3,3)
#         print(grid_f)
        
        alpha = 30.
        beta = 90.
        gamma = 90.
        q2 = Question2()
        grid_f = q2.rot_matrix(alpha, beta, gamma)
        
        print('hi')

        
        # constants
        L = 51
        N = 25
        complex_i = np.complex128(0-1j)
#         grid_f = grid_f.T # take the transpose to easily access points

        vol_f = np.zeros(3,dtype=np.complex128)

        for i in range(len(grid_f)):
            # grab (k_x,k_y) pair
            k_x = grid_f[0,i]
            k_y = grid_f[1,i]
            k_z = grid_f[2,i]

            # create the mesh grid
            x = np.linspace(-N, N, L)
            y = np.linspace(-N, N, L)
            z = np.linspace(-N, N, L)
            xv, yv, zv = np.meshgrid(x,y,z)

            # formula
            argument  = complex_i * ((2*np.pi)/L) * (xv*k_x + yv*k_y + zv*k_z)
            exp = np.exp(argument)
            ft  = np.sum(np.sum(np.sum(vol*exp)))
#             print(ft)
            vol_f[i] = ft
            print(vol_f)
        
        return vol_f

    def gen_projection(self, vol, R_theta):
        vol_sz = vol.shape[0]
        # grid on which to compute the fourier transform
        xy_plane_rot = None

        # Put your code here
        
        # create the xy plane
        num_bins = 11
        regular_grid = np.linspace(-1, 1, num_bins)
        x_grid, y_grid = np.meshgrid(regular_grid, regular_grid)
        points = np.concatenate((np.reshape(x_grid, [1, -1]),
                                 np.reshape(y_grid, [1, -1]),
                                 np.zeros((1, num_bins**2))), # the z coordinate is 0
                                axis=0)
        
        q2 = Question2()
        R_z = q2.rotate_matrix_z(R_theta)
        points_rot_z = q2.rotate_point(points, R_z)


        # Don't change the rest of the code and the output!
        ft_vol = self.nudft3(vol, xy_plane_rot)
        ft_vol = np.reshape(ft_vol, [vol_sz, vol_sz])
        proj_img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ft_vol)))
        proj_img = np.real(proj_img)
        return proj_img

    def apply_ctf(self, img, ctf):
        # Nothing to add here!
        fm = np.fft.fftshift(np.fft.fftn(img))
        cm = np.real(np.fft.ifftn(np.fft.ifftshift(fm * ctf)))
        return cm
