using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;


//ī�޶��� ������ -90���� 90���̷�
//�ϴ� ù��° ��ǥ��
//ī�޶� �������� �����̸鼭
//����⸦ �ٴڿ� ���̸鼭
//������� ������ �������� �����̴� ������ �غ���
public class robotAgent : Agent
{
    public GameObject camera;
    public GameObject Line;
    public override void CollectObservations(VectorSensor sensor)
    {

    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {

    }

    public override void OnEpisodeBegin()
    {

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var DiscreteActionsout = actionsOut.DiscreteActions;
        DiscreteActionsout[0] = 0;
        if (Input.GetKey(KeyCode.N))
        {
            DiscreteActionsout[0] = 1;
        }
    }
}
