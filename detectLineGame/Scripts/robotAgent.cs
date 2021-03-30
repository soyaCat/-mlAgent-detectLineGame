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
    const int mode = 0;
    const int line_up = 1;
    const int line_down = 2;
    const int line_left = 3;
    const int line_right = 4;
    const int line_turn_left = 5;
    const int line_turn_right = 6;
    const float stadium_horizontal = 5f;
    const float stadium_vertical = 5f;
    const float line_max_z_lot = 90f;//90
    const float line_min_z_lot = 270f;//-90
    const float line_max_y_pos = 60;
    const float line_min_y_pos = -60f;
    const float line_max_x_pos = 60;
    const float line_min_x_pos = -60f;
    const float line_lot_unit = 5f;
    const float line_move_unit = 3f;

    public GameObject camera_pack;
    public GameObject camera_target;
    public GameObject RedLine;

    public GameObject redLine;
    public GameObject B0Line;
    public GameObject B1Line;
    public GameObject B2Line;
    public GameObject B3Line;
    public GameObject Linecenter;

    private List<string> cam_target_move_mode_list = new List<string>();
    private RectTransform redLine_rectTransform;

    private RectTransform blackLine0_rectTransform;
    private RectTransform blackLine1_rectTransform;
    private RectTransform blackLine2_rectTransform;
    private RectTransform blackLine3_rectTransform;

    private void Start()
    {
        redLine_rectTransform = RedLine.GetComponent<RectTransform>();
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
            var DA_Line_up = actionBuffers.DiscreteActions[line_up];
            var DA_Line_down = actionBuffers.DiscreteActions[line_down];
            var DA_Line_left = actionBuffers.DiscreteActions[line_left];
            var DA_Line_right = actionBuffers.DiscreteActions[line_right];
            var DA_Line_turn_left = actionBuffers.DiscreteActions[line_turn_left];
            var DA_Line_turn_right = actionBuffers.DiscreteActions[line_turn_right];
            var linePosy = redLine_rectTransform.anchoredPosition.y;
            var linePosx = redLine_rectTransform.anchoredPosition.x;
            var lineRot = Mathf.Round(redLine_rectTransform.eulerAngles.z);
            if (DA_Line_up == 1)
            {
                if(linePosy < line_max_y_pos)
                    redLine_rectTransform.anchoredPosition = new Vector3(linePosx, linePosy+line_move_unit, 0f);
            }
            else if(DA_Line_down == 1)
            {
                if (linePosy > line_min_y_pos)
                    redLine_rectTransform.anchoredPosition = new Vector3(linePosx, linePosy-line_move_unit, 0f);
            }
            else if (DA_Line_left == 1)
            {
                if (linePosx > line_min_x_pos)
                    redLine_rectTransform.anchoredPosition = new Vector3(linePosx - line_move_unit, linePosy, 0f);
            }
            else if (DA_Line_right == 1)
            {
                if (linePosx < line_max_x_pos)
                    redLine_rectTransform.anchoredPosition = new Vector3(linePosx + line_move_unit, linePosy, 0f);
            }
            else if (DA_Line_turn_left == 1)
            {
                if (lineRot > line_min_z_lot || lineRot < line_max_z_lot)
                    redLine_rectTransform.eulerAngles += new Vector3(0f, 0f, line_lot_unit);
                else if (lineRot == line_min_z_lot)
                    redLine_rectTransform.eulerAngles += new Vector3(0f, 0f, line_lot_unit);

            }
            else if (DA_Line_turn_right == 1)
            {
                if (lineRot > line_min_z_lot || lineRot < line_max_z_lot)
                    redLine_rectTransform.eulerAngles -= new Vector3(0f, 0f, line_lot_unit);
                else if (lineRot == line_max_z_lot)
                    redLine_rectTransform.eulerAngles -= new Vector3(0f, 0f, line_lot_unit);
            }
        }

        B0Line.transform.position = redLine.transform.position;
        B0Line.transform.rotation = redLine.transform.rotation;

        B1Line.transform.position = redLine.transform.position;
        B1Line.transform.rotation = redLine.transform.rotation;

        B2Line.transform.position = redLine.transform.position;
        B2Line.transform.rotation = redLine.transform.rotation;

        B3Line.transform.position = redLine.transform.position;
        B3Line.transform.rotation = redLine.transform.rotation;

        Linecenter.transform.position = redLine.transform.position;
        Linecenter.transform.rotation = redLine.transform.rotation;
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
        DiscreteActionsout[mode] = 0;
        DiscreteActionsout[line_up] = 0;
        DiscreteActionsout[line_down] = 0;
        DiscreteActionsout[line_left] = 0;
        DiscreteActionsout[line_right] = 0;
        DiscreteActionsout[line_turn_left] = 0;
        DiscreteActionsout[line_turn_right] = 0;
        if (Input.GetKey(KeyCode.N))
        {
            DiscreteActionsout[mode] = 1;
            DiscreteActionsout[line_up] = 0;
            DiscreteActionsout[line_down] = 0;
            DiscreteActionsout[line_left] = 0;
            DiscreteActionsout[line_right] = 0;
            DiscreteActionsout[line_turn_left] = 0;
            DiscreteActionsout[line_turn_right] = 0;
        }
        if(Input.GetKey(KeyCode.W))
        {
            DiscreteActionsout[mode] = 2;
            DiscreteActionsout[line_up] = 1;
            DiscreteActionsout[line_down] = 0;
            DiscreteActionsout[line_left] = 0;
            DiscreteActionsout[line_right] = 0;
            DiscreteActionsout[line_turn_left] = 0;
            DiscreteActionsout[line_turn_right] = 0;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            DiscreteActionsout[mode] = 2;
            DiscreteActionsout[line_up] = 0;
            DiscreteActionsout[line_down] = 1;
            DiscreteActionsout[line_left] = 0;
            DiscreteActionsout[line_right] = 0;
            DiscreteActionsout[line_turn_left] = 0;
            DiscreteActionsout[line_turn_right] = 0;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            DiscreteActionsout[mode] = 2;
            DiscreteActionsout[line_up] = 0;
            DiscreteActionsout[line_down] = 0;
            DiscreteActionsout[line_left] = 1;
            DiscreteActionsout[line_right] = 0;
            DiscreteActionsout[line_turn_left] = 0;
            DiscreteActionsout[line_turn_right] = 0;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            DiscreteActionsout[mode] = 2;
            DiscreteActionsout[line_up] = 0;
            DiscreteActionsout[line_down] = 0;
            DiscreteActionsout[line_left] = 0;
            DiscreteActionsout[line_right] = 1;
            DiscreteActionsout[line_turn_left] = 0;
            DiscreteActionsout[line_turn_right] = 0;
        }
        else if (Input.GetKey(KeyCode.Q))
        {
            DiscreteActionsout[mode] = 2;
            DiscreteActionsout[line_up] = 0;
            DiscreteActionsout[line_down] = 0;
            DiscreteActionsout[line_left] = 0;
            DiscreteActionsout[line_right] = 0;
            DiscreteActionsout[line_turn_left] = 1;
            DiscreteActionsout[line_turn_right] = 0;
        }
        else if (Input.GetKey(KeyCode.E))
        {
            DiscreteActionsout[mode] = 2;
            DiscreteActionsout[line_up] = 0;
            DiscreteActionsout[line_down] = 0;
            DiscreteActionsout[line_left] = 0;
            DiscreteActionsout[line_right] = 0;
            DiscreteActionsout[line_turn_left] = 0;
            DiscreteActionsout[line_turn_right] = 1;
        }


    }
}
