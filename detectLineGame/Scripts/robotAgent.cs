using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;


public class robotAgent : Agent
{
    const float stadium_horizontal = 5f;
    const float stadium_vertical = 5f;
    const float line_max_z_lot = 90f;//90
    const float line_min_z_lot = 270f;//-90
    const float line_max_y_pos = 200;
    const float line_min_y_pos = -200f;
    const float line_max_x_pos = 200;
    const float line_min_x_pos = -200f;
    const float line_lot_unit = 5f;
    const float line_move_unit = 10f;

    public GameObject camera_pack;
    public GameObject camera_target;
    public GameObject RedLine;
    public GameObject BlackLine;

    private List<string> cam_target_move_mode_list = new List<string>();
    private RectTransform redLine_rectTransform;
    private RectTransform blackLine_rectTransform;

    private void Start()
    {
        redLine_rectTransform = RedLine.GetComponent<RectTransform>();
        blackLine_rectTransform = BlackLine.GetComponent<RectTransform>();
    }

    public override void CollectObservations(VectorSensor sensor)
    {

    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var act0 = actionBuffers.DiscreteActions[0];
        var nextMode = 0;

        switch (act0)
        {
            case 1:
                nextMode = 1;//카메라 타겟 오브젝트가 모서리를 따라 움직임
                break;
            case 2:
                nextMode = 2;//추정 라인을 조작함
                break;
        }

        if(nextMode == 1)
        {
            Move_Init_setting();
        }

        if(nextMode == 2)
        {
            var act1 = actionBuffers.DiscreteActions[1];
            var linePosy = redLine_rectTransform.anchoredPosition.y;
            var linePosx = redLine_rectTransform.anchoredPosition.x;
            var lineRot = Mathf.Round(redLine_rectTransform.eulerAngles.z);
            if (act1==8)
            {
                if(linePosy < line_max_y_pos)
                    redLine_rectTransform.anchoredPosition = new Vector3(linePosx, linePosy+line_move_unit, 0f);
            }
            else if(act1 == 2)
            {
                if (linePosy > line_min_y_pos)
                    redLine_rectTransform.anchoredPosition = new Vector3(linePosx, linePosy-line_move_unit, 0f);
            }
            else if (act1 == 6)
            {
                if (linePosx < line_max_x_pos)
                    redLine_rectTransform.anchoredPosition = new Vector3(linePosx + line_move_unit, linePosy, 0f);
            }
            else if (act1 == 4)
            {
                if (linePosx > line_min_x_pos)
                    redLine_rectTransform.anchoredPosition = new Vector3(linePosx - line_move_unit, linePosy, 0f);
            }
            else if(act1 == 7)
            {
                if(lineRot > line_min_z_lot || lineRot < line_max_z_lot)
                    redLine_rectTransform.eulerAngles += new Vector3(0f, 0f, line_lot_unit);
                else if(lineRot == line_min_z_lot)
                    redLine_rectTransform.eulerAngles += new Vector3(0f, 0f, line_lot_unit);

            }
            else if (act1 == 9)
            {
                if (lineRot > line_min_z_lot || lineRot < line_max_z_lot)
                    redLine_rectTransform.eulerAngles -= new Vector3(0f, 0f, line_lot_unit);
                else if(lineRot==line_max_z_lot)
                    redLine_rectTransform.eulerAngles -= new Vector3(0f, 0f, line_lot_unit);
            }
            blackLine_rectTransform.position = redLine_rectTransform.position;
            blackLine_rectTransform.rotation = redLine_rectTransform.rotation;
        }
    }

    public override void OnEpisodeBegin()
    {
        cam_target_move_mode_list.Clear();
        cam_target_move_mode_list.Add("fix_horizen");
        cam_target_move_mode_list.Add("fix_vertical");
        Move_Init_setting();
    }

    public void Move_Init_setting()
    {
        var randomInt_move_target_mode = Random.Range(0, cam_target_move_mode_list.Count());
        if (randomInt_move_target_mode == 0)
        {
            var random_figure_for_vertical = Random.Range(-(stadium_vertical / 2f), +(stadium_vertical / 2f));
            camera_target.transform.position = new Vector3(random_figure_for_vertical, 0f, 2.5f);
        }
        else
        {
            var random_figure_for_horizental = Random.Range(-(stadium_horizontal / 2f), +(stadium_horizontal / 2f));
            camera_target.transform.position = new Vector3(2.5f, 0f, random_figure_for_horizental);
        }

        this.transform.position = new Vector3(Random.Range(-(stadium_vertical / 2f), +(stadium_vertical / 2f)), 0f, Random.Range(-(stadium_horizontal / 2f), +(stadium_horizontal / 2f)));
        camera_pack.transform.LookAt(camera_target.transform.position);
        var random_x_lot_for_camera_pack = new Vector3(Random.Range(-30f, 30f), 0f, 0f);
        camera_pack.transform.eulerAngles = camera_pack.transform.eulerAngles + random_x_lot_for_camera_pack;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var DiscreteActionsout = actionsOut.DiscreteActions;
        DiscreteActionsout[0] = 0;
        DiscreteActionsout[1] = 0;
        if (Input.GetKey(KeyCode.N))
        {
            DiscreteActionsout[0] = 1;
        }
        if(Input.GetKey(KeyCode.W))
        {
            DiscreteActionsout[0] = 2;
            DiscreteActionsout[1] = 8;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            DiscreteActionsout[0] = 2;
            DiscreteActionsout[1] = 2;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            DiscreteActionsout[0] = 2;
            DiscreteActionsout[1] = 4;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            DiscreteActionsout[0] = 2;
            DiscreteActionsout[1] = 6;
        }
        else if (Input.GetKey(KeyCode.Q))
        {
            DiscreteActionsout[0] = 2;
            DiscreteActionsout[1] = 7;
        }
        else if (Input.GetKey(KeyCode.E))
        {
            DiscreteActionsout[0] = 2;
            DiscreteActionsout[1] = 9;
        }


    }
}
