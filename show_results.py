import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_experiment_information(model, x, st_observation_list, st_prediction_list, xt_prediction_list, position):
    if not model.training:
        origin_total_dim = model.total_dim
        model.total_dim = 512

    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    sample_id = np.random.randint(0, model.batch_size, size=(1))
    sample_imgs = x[sample_id]

    st_observation_sample = np.zeros((model.observe_dim, model.s_dim))
    for t in range(model.observe_dim):
        st_observation_sample[t] = st_observation_list[t][sample_id].cpu().detach().numpy()

    st_prediction_sample = np.zeros((model.total_dim - model.observe_dim, model.s_dim))
    for t in range(model.total_dim - model.observe_dim):
        st_prediction_sample[t] = st_prediction_list[t][sample_id].cpu().detach().numpy()

    st_2_max = np.maximum(np.max(st_observation_sample[:, 0]), np.max(st_prediction_sample[:, 0]))
    st_2_min = np.minimum(np.min(st_observation_sample[:, 0]), np.min(st_prediction_sample[:, 0]))
    st_1_max = np.maximum(np.max(st_observation_sample[:, 1]), np.max(st_prediction_sample[:, 1]))
    st_1_min = np.minimum(np.min(st_observation_sample[:, 1]), np.min(st_prediction_sample[:, 1]))
    axis_st_1_max = st_1_max + (st_1_max - st_1_min) / 10.0
    axis_st_1_min = st_1_min - (st_1_max - st_1_min) / 10.0
    axis_st_2_max = st_2_max + (st_2_max - st_2_min) / 10.0
    axis_st_2_min = st_2_min - (st_2_max - st_2_min) / 10.0

    fig = plt.figure()
    # interaction mode
    plt.ion()

    # observation phase
    for t in range(240, model.observe_dim):
        position_h_t = np.asscalar(position[sample_id, 0, t])
        position_w_t = np.asscalar(position[sample_id, 1, t])
        sample_imgs_t = np.copy(sample_imgs.cpu().detach().numpy())
        observed_img = np.copy(sample_imgs[:, :, 3 * position_h_t: 3 * position_h_t + 8,
                               3 * position_w_t: 3 * position_w_t + 8].cpu().detach().numpy())

        sample_imgs_t[0, 0, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 1.0
        sample_imgs_t[0, 0, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 1.0
        sample_imgs_t[0, 0, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 1.0
        sample_imgs_t[0, 0, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 1.0
        sample_imgs_t[0, 1, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 0.0
        sample_imgs_t[0, 1, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
        sample_imgs_t[0, 1, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        sample_imgs_t[0, 1, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        sample_imgs_t[0, 2, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 0.0
        sample_imgs_t[0, 2, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
        sample_imgs_t[0, 2, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        sample_imgs_t[0, 2, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 0.0

        fig.clf()

        plt.suptitle('t = ' + str(t) + '\n' + 'OBSERVATION PHASE', fontsize=25)

        gs = gridspec.GridSpec(20, 20)

        # subfigure 1
        ax1 = plt.subplot(gs[1:10, 1:10])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        plt.axis('off')
        plt.imshow(sample_imgs_t.reshape([3, 32, 32]).transpose((1, 2, 0)))

        # subfigure 2
        ax2 = plt.subplot(gs[4:8, 11:15])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_aspect('equal')
        ax2.set_title('Observation')
        plt.axis('off')
        plt.imshow(observed_img.reshape([3, 8, 8]).transpose((1, 2, 0)))

        # subfigure 3

        # subfigure 4
        ax4 = plt.subplot(gs[11:20, 1:10])
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_title('True states')
        ax4.set_aspect('equal')
        plt.axis([-1, 9, -1, 9])
        plt.gca().invert_yaxis()
        plt.plot(position[sample_id, 1, 0: t + 1].T, position[sample_id, 0, 0: t + 1].T, color='k',
                 linestyle='solid', marker='o')
        plt.plot(position[sample_id, 1, t], position[sample_id, 0, t], 'bs')

        # subfigure 5
        ax5 = plt.subplot(gs[11:20, 11:20])
        ax5.set_xlabel('$s_1$')
        ax5.set_ylabel('$s_2$')
        ax5.set_title('Inferred states')
        plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])
        plt.gca().invert_yaxis()
        plt.plot(st_observation_sample[0: t + 1, 1], st_observation_sample[0: t + 1, 0], color='k',
                 linestyle='solid', marker='o')
        plt.plot(st_observation_sample[t, 1], st_observation_sample[t, 0], 'bs')

        plt.pause(0.01)

    # predition phase
    for t in range(model.total_dim - model.observe_dim):
        position_h_t = np.asscalar(position[sample_id, 0, t + model.observe_dim])
        position_w_t = np.asscalar(position[sample_id, 1, t + model.observe_dim])
        sample_imgs_t = np.copy(sample_imgs.cpu().detach().numpy())
        observed_img = np.copy(sample_imgs[:, :, 3 * position_h_t: 3 * position_h_t + 8,
                               3 * position_w_t: 3 * position_w_t + 8].cpu().detach().numpy())
        predict_img = xt_prediction_list[t][np.asscalar(sample_id)].cpu().detach().numpy()

        sample_imgs_t[0, 0, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 1.0
        sample_imgs_t[0, 0, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 1.0
        sample_imgs_t[0, 0, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 1.0
        sample_imgs_t[0, 0, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 1.0
        sample_imgs_t[0, 1, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 0.0
        sample_imgs_t[0, 1, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
        sample_imgs_t[0, 1, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        sample_imgs_t[0, 1, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        sample_imgs_t[0, 2, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 0.0
        sample_imgs_t[0, 2, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
        sample_imgs_t[0, 2, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        sample_imgs_t[0, 2, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 0.0

        fig.clf()

        plt.suptitle('t = ' + str(t + model.observe_dim) + '\n' + 'PREDICTION PHASE', fontsize=25)

        gs = gridspec.GridSpec(20, 20)
        '''
        ax1 = plt.subplot(gs[0:4, 0:4])
        ax2 = plt.subplot(gs[1:3, 4:6])
        ax3 = plt.subplot(gs[1:3, 6:8])
        ax4 = plt.subplot(gs[4:8, 0:4])
        ax5 = plt.subplot(gs[4:8, 4:8])
        '''
        # subfigure 1
        ax1 = plt.subplot(gs[1:10, 1:10])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        plt.axis('off')
        plt.imshow(sample_imgs_t.reshape([3, 32, 32]).transpose((1, 2, 0)))

        # subfigure 2
        ax2 = plt.subplot(gs[4:8, 11:15])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_aspect('equal')
        ax2.set_title('Observation')
        plt.axis('off')
        plt.imshow(observed_img.reshape([3, 8, 8]).transpose((1, 2, 0)))

        # subfigure 3
        ax3 = plt.subplot(gs[4:8, 16:20])
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_aspect('equal')
        ax3.set_title('Prediction')
        plt.axis('off')
        plt.imshow(predict_img.reshape([3, 8, 8]).transpose((1, 2, 0)))

        # subfigure 4
        ax4 = plt.subplot(gs[11:20, 1:10])
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_title('True states')
        ax4.set_aspect('equal')
        plt.axis([-1, 9, -1, 9])
        plt.gca().invert_yaxis()
        plt.plot(position[sample_id, 1, 0: model.observe_dim + 1].T,
                 position[sample_id, 0, 0: model.observe_dim + 1].T, color='k', linestyle='solid', marker='o')
        plt.plot(position[sample_id, 1, t + model.observe_dim], position[sample_id, 0, t + model.observe_dim], 'bs')

        # subfigure 5
        ax5 = plt.subplot(gs[11:20, 11:20])
        ax5.set_xlabel('$s_1$')
        ax5.set_ylabel('$s_2$')
        ax5.set_title('Inferred states')
        plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])
        plt.gca().invert_yaxis()
        plt.plot(st_observation_sample[:, 1], st_observation_sample[:, 0], color='k', linestyle='solid', marker='o')
        plt.plot(st_prediction_sample[t, 1], st_prediction_sample[t, 0], 'bs')

        plt.pause(0.01)

    # show figure
    plt.show()

    # close figure
    plt.close(fig)

    # close interaction mode
    plt.ioff()

    if not model.training:
        model.total_dim = origin_total_dim