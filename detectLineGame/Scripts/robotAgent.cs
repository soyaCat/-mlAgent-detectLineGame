using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;


//카메라의 범위는 -90에서 90사이로
//일단 첫번째 목표는
//카메라를 랜덤으로 움직이면서
//막대기를 바닥에 붙이면서
//막대기의 각도를 랜덤으로 움직이는 것으로 해보자
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
