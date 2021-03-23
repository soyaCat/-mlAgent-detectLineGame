using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class lineFollower : MonoBehaviour
{
    public GameObject redLine;
    public GameObject B0Line;
    public GameObject B1Line;
    public GameObject B2Line;
    public GameObject B3Line;
    // Start is called before the first frame update
    void Start()
    {
        B0Line.transform.position = redLine.transform.position;
        B0Line.transform.rotation = redLine.transform.rotation;

        B1Line.transform.position = redLine.transform.position;
        B1Line.transform.rotation = redLine.transform.rotation;

        B2Line.transform.position = redLine.transform.position;
        B2Line.transform.rotation = redLine.transform.rotation;

        B3Line.transform.position = redLine.transform.position;
        B3Line.transform.rotation = redLine.transform.rotation;
    }

    // Update is called once per frame
    void Update()
    {
        B0Line.transform.position = redLine.transform.position;
        B0Line.transform.rotation = redLine.transform.rotation;
        
        B1Line.transform.position = redLine.transform.position;
        B1Line.transform.rotation = redLine.transform.rotation;
        
        B2Line.transform.position = redLine.transform.position;
        B2Line.transform.rotation = redLine.transform.rotation;
        
        B3Line.transform.position = redLine.transform.position;
        B3Line.transform.rotation = redLine.transform.rotation;
    }
}
