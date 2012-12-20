#include <iostream>

#include "HGroup.h"
#include "HDataset.h"
#include "HIntAttribute.h"
#include "HFloatAttribute.h"
#include "hdf5_hl.h"
#define MAX_NAME 1024



void
do_dtype(hid_t tid) {

	H5T_class_t t_class;
	t_class = H5Tget_class(tid);
	if(t_class < 0){ 
		puts(" Invalid datatype.\n");
	} else {
		/* 
		 * Each class has specific properties that can be 
		 * retrieved, e.g., size, byte order, exponent, etc. 
		 */
		if(t_class == H5T_INTEGER) {
		      puts(" Datatype is 'H5T_INTEGER'.\n");
			/* display size, signed, endianess, etc. */
		} else if(t_class == H5T_FLOAT) {
		      puts(" Datatype is 'H5T_FLOAT'.\n");
			/* display size, endianess, exponennt, etc. */
		} else if(t_class == H5T_STRING) {
		      puts(" Datatype is 'H5T_STRING'.\n");
			/* display size, padding, termination, etc. */
		} else if(t_class == H5T_BITFIELD) {
		      puts(" Datatype is 'H5T_BITFIELD'.\n");
			/* display size, label, etc. */
		} else if(t_class == H5T_OPAQUE) {
		      puts(" Datatype is 'H5T_OPAQUE'.\n");
			/* display size, etc. */
		} else if(t_class == H5T_COMPOUND) {
		      puts(" Datatype is 'H5T_COMPOUND'.\n");
			/* recursively display each member: field name, type  */
		} else if(t_class == H5T_ARRAY) {
		      puts(" Datatype is 'H5T_COMPOUND'.\n");
			/* display  dimensions, base type  */
		} else if(t_class == H5T_ENUM) {
		      puts(" Datatype is 'H5T_ENUM'.\n");
			/* display elements: name, value   */
		} else  {
		      puts(" Datatype is 'Other'.\n");
		      /* eg. Object Reference, ...and so on ... */
		}
	}
}


void do_attr(hid_t aid) {
	ssize_t len;
	hid_t atype;
	hid_t aspace;
	char buf[MAX_NAME]; 

	/* 
	 * Get the name of the attribute.
	 */
	len = H5Aget_name(aid, MAX_NAME, buf );
	printf("    Attribute Name : %s\n",buf);

	/*    
	 * Get attribute information: dataspace, data type 
	 */
	aspace = H5Aget_space(aid); /* the dimensions of the attribute data */

	atype  = H5Aget_type(aid); 
	do_dtype(atype);

	/*
	 * The datatype and dataspace can be used to read all or
	 * part of the data.  (Not shown in this example.)
	 */

	  /* ... read data with H5Aread, write with H5Awrite, etc. */

	H5Tclose(atype);
	H5Sclose(aspace);
}

void
scan_attrs(hid_t oid) {
	int na;
	hid_t aid;
	int i;
	
	na = H5Aget_num_attrs(oid);

	for (i = 0; i < na; i++) {
		aid =	H5Aopen_idx(oid, (unsigned int)i );
		do_attr(aid);
		H5Aclose(aid);
	}
}

void 
do_dset(hid_t did)
{
	hid_t tid;
	hid_t pid;
	hid_t sid;
	hsize_t size;
	char ds_name[MAX_NAME];

        /*
         * Information about the group:
         *  Name and attributes
         *
         *  Other info., not shown here: number of links, object id
         */
	H5Iget_name(did, ds_name, MAX_NAME  );
	printf("Dataset Name : ");
	puts(ds_name);
	printf("\n");

	/*
	 *  process the attributes of the dataset, if any.
	 */
	scan_attrs(did);
  
	/*    
	 * Get dataset information: dataspace, data type 
	 */
	sid = H5Dget_space(did); /* the dimensions of the dataset (not shown) */
	tid = H5Dget_type(did);
	printf(" DATA TYPE:\n");
	do_dtype(tid);

	/*
	 * Retrieve and analyse the dataset properties
	 */
	pid = H5Dget_create_plist(did); /* get creation property list */
	//do_plist(pid);
	size = H5Dget_storage_size(did);
	printf("Total space currently written in file: %d\n",(int)size);
	
	int rank      = H5Sget_simple_extent_ndims(sid);
	
	std::cout<<"rank "<<rank<<"\n";
	hsize_t     dims_out[2];
	int status_n  = H5Sget_simple_extent_dims(sid, dims_out, NULL);
    printf(" dimensions %lu x %lu \n",
	   (unsigned long)(dims_out[0]), (unsigned long)(dims_out[1]));
	   
	hid_t plist = H5Dget_create_plist (did);
	
	int numfilt = H5Pget_nfilters (plist);
    printf ("Number of filters associated with dataset: %i\n", numfilt);
	
	for (int i=0; i<numfilt; i++) {
       size_t nelmts = 0;
	   unsigned flags, filter_info;
       H5Z_filter_t filter_type = H5Pget_filter (plist, 0, &flags, &nelmts, NULL, 0, NULL,
                     &filter_info);
       printf ("Filter Type: ");
       switch (filter_type) {
         case H5Z_FILTER_DEFLATE:
              printf ("H5Z_FILTER_DEFLATE\n");
              break;
         case H5Z_FILTER_SZIP:
              printf ("H5Z_FILTER_SZIP\n");
              break;
         default:
              printf ("Other filter type included.\n");
         }
    }

	/*
	float *data = new float[dims_out[0]*dims_out[1]];
	herr_t status = H5Dread(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
                    data);
					
	if(status < 0)
		printf("read data error\n");
		
	for(int i=0; i < dims_out[0]; i++)
		printf("(%f %f %f) ", data[i*3], data[i*3+1], data[i*3+2]);
	//	
	delete[] data;
	*/
	/*
	 * The datatype and dataspace can be used to read all or
	 * part of the data.  (Not shown in this example.)
	 */

	  /* ... read data with H5Dread, write with H5Dwrite, etc. */

	H5Pclose(pid);
	H5Tclose(tid);
	H5Sclose(sid);
}

void 
do_link(hid_t gid, char *name) {
	herr_t status;
	char target[MAX_NAME];

	status = H5Gget_linkval(gid, name, MAX_NAME, target  ) ;
	printf("Symlink: %s points to: %s\n", name, target);
}

void
scan_group(hid_t gid) {
	int i;
	ssize_t len;
	hsize_t nobj;
	herr_t err;
	int otype;
	hid_t grpid, typid, dsid;
	char group_name[MAX_NAME];
	char memb_name[MAX_NAME];

        /*
         * Information about the group:
         *  Name and attributes
         *
         *  Other info., not shown here: number of links, object id
         */
	len = H5Iget_name (gid, group_name, MAX_NAME);

	printf("Group Name: %s\n",group_name);
	
	scan_attrs(gid);
	
	err = H5Gget_num_objs(gid, &nobj);
	
	if(nobj > 0)
		printf("Group %s has %d sub-objs\n",group_name, nobj);
	for (i = 0; i < nobj; i++) {
		//printf("  Member Id: %d ",i);//fflush(stdout);
		len = H5Gget_objname_by_idx(gid, (hsize_t)i, 
			memb_name, (size_t)MAX_NAME );
			
			//printf("   %d ",len);//fflush(stdout);
		printf("  Member Name: %s ",memb_name);//fflush(stdout);
		otype =  H5Gget_objtype_by_idx(gid, (size_t)i );
		
		switch(otype) {
			case H5G_LINK:
				printf("\n_  SYM_LINK:\n");
				do_link(gid,memb_name);
				break;
			case H5G_GROUP:
				printf("\n_  GROUP:\n");
				grpid = H5Gopen(gid,memb_name, H5P_DEFAULT);
				scan_group(grpid);
				H5Gclose(grpid);
				break;
			case H5G_DATASET:
				printf("\n_  DATASET:\n");
				dsid = H5Dopen(gid,memb_name, H5P_DEFAULT);
				do_dset(dsid);
				H5Dclose(dsid);
				break;
			case H5G_TYPE:
				printf("\n_  DATA TYPE:\n");
				typid = H5Topen(gid,memb_name, H5P_DEFAULT);
				do_dtype(typid);
				H5Tclose(typid);
				break;
			default:
				printf("\n_ unknown?\n");
				break;
			}

		
	}
}

char diagnoseFile(const char* filename)
{
	hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	hid_t    grp = H5Gopen(file_id,"/", H5P_DEFAULT);
	scan_group(grp);
	H5Gclose(grp);
	herr_t   status = H5Fclose(file_id);
	if(status < 0)
		printf("diagnose not closed\n");
	
	return 1;
}

char create_group(hid_t file, const char * name)
{
	if(H5LTpath_valid(file, name, 1)) {
		printf("%s exists\n", name);
		return 1;
	}
	  
	hid_t group_id = H5Gcreate(file, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	herr_t status = H5Gclose (group_id);
	return 1;
}

char writeFile(const char* filename, const char * dsetname)
{
	hid_t       dataspace, dataset; 
	hsize_t     dims[2];
	dims[0] = 4; 
   dims[1] = 6; 
   int         data[4][6];          /* data to write */
    int         i, j;
   for(j = 0; j < 4; j++)
	for(i = 0; i < 6; i++)
	    data[j][i] = i + j;
	
	hid_t       file_id;   /* file identifier */
      herr_t      status;

      /* Create a new file using default properties. */
	  file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
	  
	  if(file_id < 0) {
		printf("new file!\n");
		file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	  }
	  
	  create_group(file_id, "/boo");
	  
	  dataspace = H5Screate_simple(2, dims, NULL);
	  
	  htri_t dset_valid = H5LTpath_valid(file_id, dsetname, 1);
	  
	  if(!dset_valid)
			printf("%s is not valid\n", dsetname);
	  
	  dataset = H5Dopen2(file_id, dsetname, H5P_DEFAULT);
	  
	  if(dataset < 0) {
		printf("new data set!\n");
		dataset = H5Dcreate(file_id, dsetname, H5T_NATIVE_INT, dataspace, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	  }
	  
	  
	  
						  
	status = H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

	H5Sclose(dataspace);
	H5Dclose(dataset);
      /* Terminate access to the file. */
      status = H5Fclose(file_id); 
	return 1;
}

char readFile(const char* filename)
{
	hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	hid_t dataset = H5Dopen2(file_id, "/foo2", H5P_DEFAULT);
	hid_t datatype = H5Dget_type(dataset);
	H5T_class_t t_class = H5Tget_class(datatype);
	
	if(t_class == H5T_INTEGER)
		std::cout<<"data set has native int type \n";
	else
		std::cout<<"data set has unknown type \n";
		
	size_t      size = H5Tget_size(datatype);
	std::cout<<"data size "<<size<<"\n";
	
	hid_t dataspace = H5Dget_space(dataset);    /* dataspace handle */
	int rank      = H5Sget_simple_extent_ndims(dataspace);
	
	std::cout<<"rank "<<rank<<"\n";
	hsize_t     dims_out[2];
	int status_n  = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
    printf(" dimensions %lu x %lu \n",
	   (unsigned long)(dims_out[0]), (unsigned long)(dims_out[1]));
	   
	int         data[4][6];
	herr_t status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
                    data);
	int i, j;				
	for(j = 0; j < 4; j++) {
	for(i = 0; i < 6; i++) {
		std::cout<<" "<<data[j][i];
	}
		std::cout<<" \n";
	}
	
	H5Tclose(datatype);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Fclose(file_id);
	return 1;
}

int main (int argc, char * const argv[]) {
    // insert code here...
    std::cout << "Hello, HDF5!\n";
	
	//writeFile("dset.h5","/car");
	//writeFile("dset.h5","/foo2");
	//writeFile("dset.h5","/boo/tar");
	//writeFile("dset.h5","/boo/c");
	//writeFile("dset.h5","/boo/x");
	//diagnoseFile("/Users/jianzhang/Desktop/scene.abc");
	
	//
	
	if(HObject::FileIO.open("dset.h5", HDocument::oReadAndWrite)) {
		printf("opened");
	}
	else {
		printf("not opened");
	}
	
	HGroup grpA("/A1");
	grpA.create();
	
	if(HObject::FileIO.checkExist("/A1"))
		printf("find grp /A1");
		
	HIntAttribute rootAttr("/.range");
	rootAttr.create(4);
	rootAttr.open();
	
	int vrange[4];
	vrange[0] = 31;
	vrange[1] = 1037;
	vrange[2] = -87;
	vrange[3] = 7;
	if(!rootAttr.write(vrange)) std::cout<<"/.range write failed\n";
	rootAttr.close();
	
	HFloatAttribute fltAttr("/.time");
	fltAttr.create(2);
	fltAttr.open();
	
	float vtime[2];
	vtime[0] = .00947;
	vtime[1] = -36.450;
	if(!fltAttr.write(vtime)) std::cout<<"/.time write failed\n";
	fltAttr.close();
	
	HGroup grpAC("/A1/C");
	grpAC.create();
	
	HGroup grpB("/B2");
	grpB.create();
	
	HGroup grpBD("/B2/D");
	grpBD.create();
	
	HGroup grpBDE("/B2/D/E");
	grpBDE.create();
	
	HDataset dsetAg("/A1/g");
	dsetAg.create(32,1);
	dsetAg.open();
	dsetAg.write();
	dsetAg.close();
	
	HDataset dsetBg("/B2/D/g");
	dsetBg.create(32,1);
	dsetBg.open();
	dsetBg.write();
	dsetBg.close();
	
	HObject::FileIO.close();

	HObject::FileIO.open("dset.h5", HDocument::oReadAndWrite);
	dsetAg.open();
	dsetAg.read();
	dsetAg.close();
	
	dsetBg.open();
	dsetBg.read();
	dsetBg.close();
	
	HObject::FileIO.deleteObject("/A1");
	HObject::FileIO.close();
	//printf("diagnose\n");
	
	diagnoseFile("dset.h5");
	//diagnoseFile("/Users/jianzhang/man/bakep/alot/hdt.h5");
	
	if(HObject::FileIO.open("dset.h5", HDocument::oReadOnly)) {
		printf("opened to read /.range\n");
	}
	
	HIntAttribute inAttr("/.range");
	if(!inAttr.open())
		std::cout<<"/.range not opened\n";
	
	int *vinattr;
	if(!inAttr.read(vinattr)) std::cout<<"./range read failed\n";
	
	int at = inAttr.dataSpaceDimension();
	//
	
	inAttr.close();
	
	std::cout<<"inst attr /.range: "<<vinattr[0]<<", "<<vinattr[1]<<", "<<vinattr[2]<<", "<<vinattr[3]<<std::endl;
	std::cout<<"dim attr: "<<at<<std::endl;
	
	HIntAttribute dummy("/.nouse");
	//if(!dummy.open()) std::cout<<"no dummy";
	
	HFloatAttribute inTime("/.time");
	inTime.open();
	
	float *vintime;
	inTime.read(vintime);
	
	std::cout<<"float attr /.time: "<<vintime[0]<<", "<<vintime[1]<<std::endl;
	
	
	inTime.close();
	HObject::FileIO.close();
	              
	return 0;
}
