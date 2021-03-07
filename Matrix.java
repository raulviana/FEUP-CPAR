import java.time.Clock;
import java.util.Scanner;  // Import the Scanner class

public class Matrix {
	public static double OnMult(int m_ar, int m_br) 
{
	
	long Time1, Time2;
	
	double temp;
	int i, j, k;
	double[] pha = new double[m_ar*m_ar*Double.BYTES], phb = new double[m_ar*m_ar*Double.BYTES], phc = new double[m_ar*m_ar*Double.BYTES];
	
	for(i=0; i<m_ar; i++)
		for(j=0; j<m_ar; j++)
			pha[i*m_ar + j] = (double)1;



	for(i=0; i<m_br; i++)
		for(j=0; j<m_br; j++)
			phb[i*m_br + j] = (double)(i+1);



    Time1 = System.currentTimeMillis();

	for(i=0; i<m_ar; i++)
	{	for( j=0; j<m_br; j++)
		{	temp = 0;
			for( k=0; k<m_ar; k++)
			{	
				temp += pha[i*m_ar+k] * phb[k*m_br+j];
			}
			phc[i*m_ar+j]=temp;
		}
	}


    Time2 = System.currentTimeMillis();
	System.out.println("Time: " + (double)(Time2 - Time1)/1000 + "seconds\n");

	System.out.println("Result matrix: ");
	for (i = 0; i < 1; i++) {
		for (j = 0; j < Math.min(10, m_br); j++)
			System.out.print(phc[j] + " ");
	}
	return (double)(Time2 - Time1);
}

public static double OnMultLine(int m_ar, int m_br)
{
	long Time1, Time2;
	
	double temp;
	int i, j, k;
	double[] pha = new double[m_ar*m_ar*Double.BYTES], phb = new double[m_ar*m_ar*Double.BYTES], phc = new double[m_ar*m_ar*Double.BYTES];
	
	
	for(i=0; i<m_ar; i++)
		for(j=0; j<m_ar; j++)
			pha[i*m_ar + j] = (double)1.0;



	for(i=0; i<m_br; i++)
		for(j=0; j<m_br; j++)
			phb[i*m_br + j] = (double)(i+1);



	Time1 = System.currentTimeMillis();

	for(i=0; i<m_ar; i++)
	{	for( k=0; k<m_ar; k++)
		{	
			for( j=0; j<m_br; j++)
			{	
				phc[i*m_ar+j]+= pha[i*m_ar+k] * phb[k*m_br+j];
			}
		}
	}

    Time2 = System.currentTimeMillis();
	System.out.println("Time: " + (double)(Time2 - Time1)/1000 + "seconds\n");
	System.out.println("Result matrix: ");
	for (i = 0; i < 1; i++) {
		for (j = 0; j < Math.min(10, m_br); j++)
			System.out.print(phc[j]);
	}

	return (double)(Time2 - Time1);
}

public static double blockMult(int m_ar, int m_br, Scanner scanner){
	int blockSize;
	System.out.println("\n****Block Multiplication****");
	System.out.println("Block size?: [0 <= Block Size <= "  + m_ar + " ]");
	blockSize = scanner.nextInt();
	if(blockSize <= 0 || blockSize > m_ar) return -1;

	long Time1, Time2;
	
	double temp;
	int i, j, k;

	double[] pha = new double[m_ar*m_ar*Double.BYTES], phb = new double[m_ar*m_ar*Double.BYTES], phc = new double[m_ar*m_ar*Double.BYTES];
	double[] blA = new double[(blockSize * blockSize) * Double.BYTES],blB = new double[(blockSize * blockSize)*Double.BYTES];

	for(i=0; i<m_ar; i++)
		for(j=0; j<m_ar; j++)
			pha[i*m_ar + j] = (double)1.0;

	for(i=0; i<m_br; i++)
		for(j=0; j<m_br; j++)
			phb[i*m_br + j] = (double)(i+1);

    Time1 = System.currentTimeMillis();

	for(int kk=0;kk<m_ar;kk+= blockSize){
			for(int jj=0;jj<m_ar;jj+= blockSize){
					for(i=0;i<m_ar;i++){
							for(k = kk; k<(kk+blockSize); k++){
									temp = 0;
								for(j = jj; j<(jj+blockSize); j++) {
											phc[i*m_ar + j] += pha[i*m_ar + k]*phb[k*m_ar +j];							
									}
							}
					}
			}
	}
	

    Time2 = System.currentTimeMillis();
	System.out.println("Time: " + (double)(Time2 - Time1)/1000 + "seconds\n");
	System.out.println("Result matrix: ");
	for (i = 0; i < 1; i++) {
		for (j = 0; j < Math.min(10, m_br); j++)
			System.out.print(phc[j]);
	}
	return (double)(Time2 - Time1);	
}

/*private float produtoInterno(float *v1, float *v2, int col)
{
	int i;
	float soma=0.0;	

	for(i=0; i<col; i++)
		soma += v1[i]*v2[i];
	
	return(soma);

}*/

	public static void main(String[] args) {
        
	char c;
	int lin, col, nt=1;
	int op = 1;
	int startSize, endSize, step;
  	//long values[2];
  	int ret;
	double time;
	//ofstream fileStream;
	Scanner myObj = new Scanner(System.in);  // Create a Scanner object

	do {
		System.out.println("1. Multiplication");
		System.out.println("2. Line Multiplication");
		System.out.println("3. Block Multiplication");
		System.out.println("Selection?: [0 to stop] ");
		op = myObj.nextInt();
		if (op == 0){
			System.out.println("Exiting");
			break;
		}
	
		System.out.println("Parameters?: [StartSize] [Step] [EndSize]\n ");
		String params = myObj.nextLine();
		params.split(" ");
		startSize = myObj.nextInt();
		step = myObj.nextInt();
		endSize = myObj.nextInt();
		// printf("Dimensions: lins cols ? ");
   		// cin >> lin >> col;

		//fileStream.open(FILENAME, fstream::app);
		System.out.println("Op" + op);
		do{
			// Start counting
			//ret = PAPI_start(EventSet);
			//if (ret != PAPI_OK) cout << "ERRO: Start PAPI" << endl;
			lin = startSize;
			col = startSize;
			switch (op){
				case 1: 
					time = OnMult(lin, col);
					break;
				case 2:
					time = OnMultLine(lin, col);
					break;
				case 3:
					time = blockMult(lin, col, myObj);
			}

			/*ret = PAPI_stop(EventSet, values);
			if (ret != PAPI_OK) cout << "ERRO: Stop PAPI" << endl;
			printf("L1 DCM: %lld \n",values[0]);
			printf("L2 DCM: %lld \n",values[1]);

			fileStream << startSize << ", " << time << ", " << values[0] << ", " << values[1] << "\n";
			ret = PAPI_reset( EventSet );
			if ( ret != PAPI_OK )
				std::cout << "FAIL reset" << endl; */
			
			System.out.println("\nCurrent size: " + startSize);
			System.out.println("*****************************\n\n");

			startSize += step;

		}while(startSize <= endSize);
		//fileStream.close();

	}while (op != 0);
	myObj.close();
		/*ret = PAPI_remove_event( EventSet, PAPI_L1_DCM );
		if ( ret != PAPI_OK )
			std::cout << "FAIL remove event" << endl; 

		ret = PAPI_remove_event( EventSet, PAPI_L2_DCM );
		if ( ret != PAPI_OK )
			std::cout << "FAIL remove event" << endl; 

		ret = PAPI_destroy_eventset( &EventSet );
		if ( ret != PAPI_OK )
			std::cout << "FAIL destroy" << endl;*/

    }

}